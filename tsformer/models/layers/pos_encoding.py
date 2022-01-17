'''
Author: jianzhnie
Date: 2022-01-17 16:38:13
LastEditTime: 2022-01-17 16:38:14
LastEditors: jianzhnie
Description:

'''

import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):

    def __init__(self,
                 emb_size: int = 512,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        if emb_size % 2 != 0:
            raise ValueError(
                'Cannot use sin/cos postional encoding with odd dim (got dim ={:d}'
                .format(emb_size))

        self.emb_size = emb_size
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2, dtype=torch.float) *
            (-math.log(10000.0) / emb_size))
        pos_embedding = torch.zeros(max_len, emb_size)
        # 偶数位置编码
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置编码
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        pos_embedding = pos_embedding.unsqueeze(0).transpose(0, 1)
        # 不对位置编码求梯度
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape  [seq_len, batch_size, embedding_dim]
        """
        x = x * math.sqrt(self.emb_size)
        # 输入的词向量与位置编码相加
        x = x + self.pos_embedding[:x.size(0), :]
        return self.dropout(x)
