'''
Author: jianzhnie
Date: 2022-01-11 17:45:54
LastEditTime: 2022-01-11 19:18:43
LastEditors: jianzhnie
Description:

'''
import torch


class TriangularCausalMask():

    def __init__(self, B, L, device='cpu'):
        mask_shape = [B, 1, L, L]
        # mask 上三角矩阵
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool),
                diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():

    def __init__(self, B, H, L, index, scores, device='cpu'):
        _mask = torch.ones(
            L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


if __name__ == '__main__':
    tm = TriangularCausalMask(3, 5)
    print(tm._mask)
