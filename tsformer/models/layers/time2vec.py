'''
Author: jianzhnie
Date: 2022-01-17 14:13:42
LastEditTime: 2022-01-17 15:05:11
LastEditors: jianzhnie
Description:

'''
import torch
from torch import nn
from torch.nn.parameter import Parameter


class Time2Vec(nn.Module):

    def __init__(self, in_features, out_features, act='sin', **kwargs):
        super(Time2Vec, self).__init__(**kwargs)
        self.wb = Parameter(torch.randn(in_features, 1))
        self.bb = Parameter(torch.randn(in_features, 1))
        self.wa = Parameter(torch.randn(in_features, out_features - 1))
        self.ba = Parameter(torch.randn(in_features, out_features - 1))

        self.act = act

    def forward(self, x, **kwargs):
        bias = torch.matmul(x, self.wb) + self.bb
        dp = torch.matmul(x, self.wa) + self.ba
        if self.act == 'sin':
            wgts = torch.sin(dp)
        elif self.act == 'cos':
            wgts = torch.cos(dp)
        else:
            raise NotImplementedError(
                'Neither since or cossine predictor activation be selected')
        return torch.cat([wgts, bias], dim=1)


if __name__ == '__main__':
    T2V = Time2Vec(1, 64, act='sin')
    out = T2V(torch.Tensor([[7]]))
    print(out)
