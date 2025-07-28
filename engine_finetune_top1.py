import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, precision_score,
    recall_score, jaccard_score, hamming_loss, multilabel_confusion_matrix
)
import seaborn as sns
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched
import util.mixup as multilabel_mixup


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()

    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True).float()

        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        loss /= accum_iter

        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr, max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    import os
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, roc_auc_score,
        average_precision_score, jaccard_score, hamming_loss,
        multilabel_confusion_matrix
    )
    from util import misc

    criterion = torch.nn.BCEWithLogitsLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    save_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    true_labels, pred_probs = [], []

    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True).float()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        metric_logger.update(loss=loss.item())  # ✅ 修复：添加这一行

        probs = torch.sigmoid(outputs)  # [B, C]
        pred_probs.extend(probs.cpu().numpy())
        true_labels.extend(targets.cpu().numpy())

    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)

    # ✅ 自适应阈值策略
    if mode == 'val':
        best_thresholds = np.zeros(num_class)
        for i in range(num_class):
            best_f1, best_t = 0, 0.5
            for t in np.arange(0.1, 0.91, 0.01):
                preds = (pred_probs[:, i] >= t).astype(int)
                f1 = f1_score(true_labels[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            best_thresholds[i] = best_t
        np.save(os.path.join(save_dir, "best_thresholds.npy"), best_thresholds)
        print("✅ 已保存最优阈值到 best_thresholds.npy")
        # ✅ 保存为 CSV 格式
        csv_path = os.path.join(save_dir, "best_thresholds.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "threshold"])
            for i, t in enumerate(best_thresholds):
                writer.writerow([i, round(t, 4)])
        print(f"✅ 最优阈值已保存为:\n  → {csv_path}\n  → {csv_path.replace('.csv', '.npy')}")

    elif mode == 'test':
        thresholds_path = os.path.join(save_dir, "best_thresholds.npy")
        if os.path.exists(thresholds_path):
            best_thresholds = np.load(thresholds_path)
            print(f"✅ 已加载阈值 {thresholds_path}")
        else:
            print("⚠️ 未找到 best_thresholds.npy，默认使用0.5阈值")
            best_thresholds = np.array([0.5] * num_class)
    else:
        best_thresholds = np.array([0.5] * num_class)

    # ✅ 应用阈值
    pred_binary = (pred_probs >= best_thresholds).astype(int)

    # ✅ Top-k 补全（避免空预测）
    for i in range(pred_binary.shape[0]):
        if pred_binary[i].sum() == 0:
            top_k = 1
            topk_indices = np.argsort(-pred_probs[i])[:top_k]
            pred_binary[i][topk_indices] = 1

    # ======== 指标计算 ========
    f1 = f1_score(true_labels, pred_binary, average='macro', zero_division=0)
    roc_auc = roc_auc_score(true_labels, pred_probs, average='macro')
    average_precision = average_precision_score(true_labels, pred_probs, average='macro')
    precision = precision_score(true_labels, pred_binary, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_binary, average='macro', zero_division=0)
    jaccard = jaccard_score(true_labels, pred_binary, average='macro', zero_division=0)
    hamming = hamming_loss(true_labels, pred_binary)
    score = (f1 + roc_auc + average_precision) / 3

    # 空预测样本统计
    num_empty = (pred_binary.sum(axis=1) == 0).sum()
    print(f"❗ 空白预测样本数: {num_empty} / {pred_binary.shape[0]}")

    # TensorBoard记录
    if log_writer:
        for name, val in zip(
            ['f1', 'roc_auc', 'average_precision', 'precision', 'recall', 'jaccard', 'hamming', 'score'],
            [f1, roc_auc, average_precision, precision, recall, jaccard, hamming, score]
        ):
            log_writer.add_scalar(f'perf/{name}', val, epoch)

    print(f'{mode} loss: {metric_logger.meters["loss"].global_avg:.4f}')
    print(f'F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Avg Precision: {average_precision:.4f}, '
          f'Hamming: {hamming:.4f}, Jaccard: {jaccard:.4f}, Score: {score:.4f}')

    # 保存评估结果
    result_csv = os.path.join(save_dir, f'metrics_{mode}.csv')
    file_exists = os.path.exists(result_csv)
    with open(result_csv, 'a', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['loss', 'f1', 'roc_auc', 'average_precision', 'precision', 'recall', 'jaccard', 'hamming', 'score'])
        writer.writerow([metric_logger.meters["loss"].global_avg, f1, roc_auc, average_precision, precision, recall, jaccard, hamming, score])

    # 保存预测与混淆矩阵（仅 test）
    if mode == 'test':
        np.save(os.path.join(save_dir, f"true_labels_{mode}.npy"), true_labels)
        np.save(os.path.join(save_dir, f"pred_labels_{mode}.npy"), pred_binary)

        cm_list = multilabel_confusion_matrix(true_labels, pred_binary)
        cols = 4
        rows = (num_class + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten()

        for i in range(num_class):
            cm = cm_list[i]
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'], ax=ax)
            ax.set_title(f'Class {i}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        for j in range(num_class, len(axes)):
            fig.delaxes(axes[j])

        cm_save_path = os.path.join(save_dir, f"confusion_matrix_multilabel_{mode}_ep{epoch}.jpg")
        plt.tight_layout()
        plt.savefig(cm_save_path, dpi=300)
        plt.close()
        print(f"✅ 混淆矩阵已保存至 {cm_save_path}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score

