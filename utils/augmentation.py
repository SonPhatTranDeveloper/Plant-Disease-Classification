"""
Author: Son Phat Tran
This file contains the code for data augmentation technique: CutMix and MixUp
"""
import numpy as np
import torch


def cut_mix_data(x, y, alpha=1.0):
    """Performs CutMix augmentation."""
    batch_size = x.size()[0]

    # Generate mixed sample
    lam = np.random.beta(alpha, alpha)

    # Get random index for mixing
    rand_index = torch.randperm(batch_size)

    # Get random box coordinates
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    # Perform mix up
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y, y[rand_index], lam


def rand_bbox(size, lam):
    """Generates random bounding box coordinates."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Get random box center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2