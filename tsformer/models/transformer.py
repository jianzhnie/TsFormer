'''
Author: jianzhnie
Date: 2022-01-17 16:36:57
LastEditTime: 2022-01-17 18:34:59
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

    def forward(self, inputs):
        # 与LSTM 处理情况类似， 输入数据是 batch * seq_length
        # 需要转换成  seq_length * batch 的格式
        inputs = torch.transpose(inputs, 0, 1)
        inputs = self.value_embedding(inputs) + self.position_embedding(inputs)
        # 根据序列长度生成 Padding Mask 矩阵
        inputs = self.dropout(inputs)
        hidden_states = self.encoder(inputs)
        # idx == 0 is for classification
        # 取第一个标记位置的输出作为分类层的输出
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)

        return output
