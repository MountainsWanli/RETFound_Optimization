import numpy as np
import torch

def multilabel_mixup(samples, targets, alpha=0.4):
    '''适配多标签 BCE 的 mixup'''
    lam = np.random.beta(alpha, alpha)
    batch_size = samples.size(0)
    index = torch.randperm(batch_size).to(samples.device)

    mixed_samples = lam * samples + (1 - lam) * samples[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]  # soft label

    return mixed_samples, mixed_targets
