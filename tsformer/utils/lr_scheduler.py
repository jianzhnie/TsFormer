'''
Author: jianzhnie
Date: 2022-01-14 17:18:27
LastEditTime: 2022-01-14 17:24:08
LastEditors: jianzhnie
Description:

'''
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def get_lr_scheduler(lr_scheduler, optimizer, epochs):
    """Learning Rate Scheduler."""
    if lr_scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=epochs, gamma=0.5)
    elif lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler
