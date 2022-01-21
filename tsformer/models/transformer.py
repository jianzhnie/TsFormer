'''
Author: jianzhnie
Date: 2022-01-17 16:36:57
LastEditTime: 2022-01-21 09:58:21
LastEditors: jianzhnie
Description:

'''

import torch
from torch import nn

from .layers.embed import TokenEmbedding
from .layers.pos_encoding import PositionalEncoding


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask


class Transformer(nn.Module):

    def __init__(self,
                 input_features,
                 input_seq_len,
                 hidden_dim,
                 output_seq_len,
                 dim_feedforward=512,
                 num_head=2,
                 num_layers=2,
                 dropout=0.1,
                 activation: str = 'relu'):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.position_embedding = PositionalEncoding(hidden_dim)
        self.value_embedding = TokenEmbedding(
            c_in=input_features, d_model=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.src_mask = generate_square_subsequent_mask(input_seq_len)
        self.tgt_mask = generate_square_subsequent_mask(output_seq_len)

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
        self.output = nn.Linear(hidden_dim, output_seq_len, bias=True)

    def forward(self, src, src_mask=None):
        # 与LSTM 处理情况类似， 输入数据是 batch * seq_length
        # 需要转换成  seq_length * batch 的格式
        src = self.value_embedding(src)
        src = torch.transpose(src, 0, 1)
        src = self.position_embedding(src)
        # 根据序列长度生成 Padding Mask 矩阵
        src = self.dropout(src)

        hidden_states = self.encoder(src, self.src_mask.to(src.device))
        output = self.decoder(src, hidden_states)
        output = self.output(output)

        return output
