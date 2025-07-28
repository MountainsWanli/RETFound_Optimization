import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        :param alpha: 缩放因子，可以是float或shape=[num_classes]的Tensor（支持每类不同权重）
        :param gamma: 聚焦因子，越大越聚焦于难分类样本
        :param reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C], logits (未过sigmoid)
        # targets: [B, C], 多标签 one-hot 编码

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # shape [B, C]
        pt = torch.exp(-bce_loss)  # sigmoid(x) for positive, 1-sigmoid(x) for negative
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
