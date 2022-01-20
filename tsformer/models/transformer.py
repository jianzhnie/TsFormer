'''
Author: jianzhnie
Date: 2022-01-17 16:36:57
LastEditTime: 2022-01-20 15:52:30
LastEditors: jianzhnie
Description:

'''

import torch
from torch import nn

from .layers.embed import PositionalEmbedding, TokenEmbedding


class Transformer(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_dim,
                 output_size,
                 dim_feedforward=512,
                 num_head=2,
                 num_layers=2,
                 dropout=0.1,
                 activation: str = 'relu'):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.position_embedding = PositionalEmbedding(hidden_dim)
        self.value_embedding = TokenEmbedding(
            c_in=input_size, d_model=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.src_mask = None

        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head,
                                                   dim_feedforward, dropout,
                                                   activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # 输出层

        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_head,
                                                   dim_feedforward, dropout,
                                                   activation)

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # 输出层
        self.output = nn.Linear(hidden_dim, output_size, bias=True)

    def forward(self, src):
        # 与LSTM 处理情况类似， 输入数据是 batch * seq_length
        # 需要转换成  seq_length * batch 的格式
        src = torch.transpose(src, 0, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.value_embedding(src) + self.position_embedding(src)
        # 根据序列长度生成 Padding Mask 矩阵
        src = self.dropout(src)
        hidden_states = self.encoder(src, self.src_mask)
        output = self.decoder(src, hidden_states)
        output = self.output(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask
