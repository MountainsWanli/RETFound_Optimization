# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


class MultiLabelImageDataset(Dataset):
    def __init__(self, csv_path, transform=None, class_map=None, num_classes=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.class_map = class_map
        self.num_classes = num_classes

        if self.class_map is not None:
            self.data['labels'] = self.data['labels'].apply(self.map_labels)

    def map_labels(self, label_str):
        # label_str: e.g. "1,2"
        label_list = list(map(int, str(label_str).split(',')))
        label_vec = [0] * self.num_classes
        for lbl in label_list:
            if lbl in self.class_map:
                label_vec[self.class_map[lbl]] = 1
        return label_vec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        labels = self.data.iloc[idx, 1]
        if isinstance(labels, str):
            labels = self.map_labels(labels)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels, dtype=torch.float32)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    csv_path = os.path.join(args.data_path, f"{is_train}.csv")

    # 读取数据集 csv
    df = pd.read_csv(csv_path)
    
    # 获取所有出现过的类别及其映射
    all_labels = set()
    for label_str in df['labels']:
        all_labels.update(map(int, str(label_str).split(',')))
    all_labels = sorted(list(all_labels))
    class_map = {old: new for new, old in enumerate(all_labels)}
    num_classes = len(class_map)
    args.nb_classes = num_classes  # 更新到 args 中

    # 构建数据集
    dataset = MultiLabelImageDataset(
        csv_path=csv_path,
        transform=transform,
        class_map=class_map,
        num_classes=num_classes
    )

    # === 计算类别频次 === #
    print("每类图片数量:")
    counts = np.zeros(num_classes)
    for row in df['labels']:
        for lbl in map(int, str(row).split(',')):
            if lbl in class_map:
                counts[class_map[lbl]] += 1
    print(dict(zip(range(num_classes), counts.astype(int))))

    # === 仅训练集返回采样器 === #
    if is_train == 'train':
        # 读取 count_all.csv 并构建类别权重
        count_path = "/home/itaer2/zxy/shixi/retfound2/datasets/mutil/gan2/count_all.csv"
        count_df = pd.read_csv(count_path)  # 包含 ['class', 'count']
        raw_class_counts = {row['class']: row['count'] for _, row in count_df.iterrows() if row['class'] in class_map}
        
        # 转换为 new class index
        class_counts = {class_map[k]: raw_class_counts[k] for k in raw_class_counts}
        total = sum(class_counts.values())
        class_weights = {k: total / v for k, v in class_counts.items()}
        class_weights_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(num_classes)], dtype=torch.float32)

        # === 计算每个样本的权重 === #
        sample_weights = []
        for label_str in df['labels']:
            labels = list(map(int, str(label_str).split(',')))
            label_vec = [0] * num_classes
            for lbl in labels:
                if lbl in class_map:
                    label_vec[class_map[lbl]] = 1
            weight = sum([class_weights_tensor[i] for i, v in enumerate(label_vec) if v == 1])
            sample_weights.append(weight)

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        return dataset, sampler

    else:
        return dataset,None


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if is_train == 'train':
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)