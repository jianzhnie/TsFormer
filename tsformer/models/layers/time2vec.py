'''
Author: jianzhnie
Date: 2022-01-17 14:13:42
LastEditTime: 2022-01-20 16:10:53
LastEditors: jianzhnie
Description:

'''
import torch
from torch import nn
from torch.nn.parameter import Parameter


class Time2Vec(nn.Module):

    def __init__(self, input_dim=6, embed_dim=512, act_function=torch.sin):
        assert embed_dim % input_dim == 0
        super(Time2Vec, self).__init__()
        self.enabled = embed_dim > 0
        if self.enabled:
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim))
            self.embed_bias = nn.parameter.Parameter(
                torch.randn(self.embed_dim))
            self.act_function = act_function

    def forward(self, x):
        if self.enabled:
            # size of x = [bs, sample, input_dim]
            x = torch.diag_embed(x)
            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias
            # size of x_affine = [bs, sample, embed_dim]
            x_affine_0, x_affine_remain = torch.split(
                x_affine, [1, self.embed_dim - 1], dim=-1)
            x_affine_remain = self.act_function(x_affine_remain)
            x_output = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            x_output = x_output.view(x_output.size(0), x_output.size(1), -1)
        else:
            x_output = x
        return x_output


class CustomTime2Vec(nn.Module):

    def __init__(self, input_dim=6, embed_dim=512, act='sin', **kwargs):
        super(Time2Vec, self).__init__(**kwargs)
        assert embed_dim % input_dim == 0
        self.wb = Parameter(torch.randn(input_dim, 1))
        self.bb = Parameter(torch.randn(input_dim, 1))
        self.wa = Parameter(torch.randn(input_dim, embed_dim))
        self.ba = Parameter(torch.randn(input_dim, embed_dim))

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
