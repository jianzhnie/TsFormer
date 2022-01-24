'''
Author: jianzhnie
Date: 2022-01-21 11:15:51
LastEditTime: 2022-01-24 15:05:51
LastEditors: jianzhnie
Description:

'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.attn import AttentionLayer, FullAttention, ProbAttention
from .layers.embed import DataEmbedding


class Informer(nn.Module):

    def __init__(self,
                 enc_in: int,
                 dec_in: int,
                 c_out: int,
                 seq_len: int,
                 label_len: int,
                 pred_len: int,
                 factor: int = 5,
                 d_model: int = 512,
                 n_heads: int = 8,
                 e_layers: int = 3,
                 d_layers: int = 2,
                 d_ffn: int = 512,
                 dropout=0.0,
                 embed='fixed',
                 freq='h',
                 activation='gelu'):

        super(Informer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.c_out = c_out
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq,
                                           dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq,
                                           dropout)
        # Attention
        enc_prob_attn = ProbAttention(False, factor, attention_dropout=dropout)

        # Encoder
        conv_layer = ConvLayer(d_model)
        encoder_norm = nn.LayerNorm(d_model)
        enc_attn_layer = AttentionLayer(enc_prob_attn, d_model, n_heads)
        encoder_layer = EncoderLayer(enc_attn_layer, d_model, d_ffn, dropout,
                                     activation)

        self.encoder = Encoder(encoder_layer, conv_layer, e_layers,
                               encoder_norm)

        # Decoder
        dec_prob_attn = ProbAttention(True, factor, attention_dropout=dropout)
        dec_full_attn = FullAttention(False, factor, attention_dropout=dropout)
        dec_attn_layer1 = AttentionLayer(dec_prob_attn, d_model, n_heads)
        dec_attn_layer2 = AttentionLayer(dec_full_attn, d_model, n_heads)
        decoder_layer = DecoderLayer(
            self_attn_layer=dec_attn_layer1,
            cross_attn_layer=dec_attn_layer2,
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(
            decoder_layer, num_layers=d_layers, norm_layer=decoder_norm)

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self,
                x_enc: torch.Tensor,
                x_mark_enc: torch.Tensor,
                x_dec: torch.Tensor,
                x_mark_dec: torch.Tensor,
                enc_self_mask=None,
                dec_self_mask=None,
                dec_enc_mask=None):
        """

        :param x_enc: The core tensor going into the model. Of dimension (batch_size, seq_len, enc_in)
        :type x_enc: torch.Tensor
        :param x_mark_enc: A tensor with the relevant datetime information. (batch_size, seq_len, n_datetime_feats)
        :type x_mark_enc: torch.Tensor
        :param x_dec: The datetime tensor information. Has dimension batch_size, seq_len, enc_in
        :type x_dec: torch.Tensor
        :param x_mark_dec: A tensor with the relevant datetime information. (batch_size, seq_len, n_datetime_feats)
        :type x_mark_dec: torch.Tensor
        :param enc_self_mask: The mask of the encoder model has size (), defaults to None
        :type enc_self_mask: [type], optional
        :param dec_self_mask: [description], defaults to None
        :type dec_self_mask: [type], optional
        :param dec_enc_mask: torch.Tensor, defaults to None
        :type dec_enc_mask: torch.Tensor, optional
        :return: Returns a PyTorch tensor of shape (batch_size, out_len, n_targets)
        :rtype: torch.Tensor
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, tgt_mask=dec_self_mask, memory_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class ConvLayer(nn.Module):

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):

    def __init__(self,
                 attention_layer,
                 d_model,
                 d_ffn=None,
                 dropout=0.1,
                 activation='relu'):
        """[summary]

        :param attention: [description]
        :type attention: [type]
        :param d_model: [description]
        :type d_model: [type]
        :param d_ff: [description], defaults to None
        :type d_ff: [type], optional
        :param dropout: [description], defaults to 0.1
        :type dropout: float, optional
        :param activation: [description], defaults to "relu"
        :type activation: str, optional
        """
        super(EncoderLayer, self).__init__()
        d_ffn = d_ffn or 4 * d_model
        self.attention = attention_layer
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ffn, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ffn, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, src, attn_mask=None):
        # x [B, L, D]
        src2 = self.attention(src, src, src, attn_mask=attn_mask)[0]

        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.dropout1(self.activation(self.conv1(src.transpose(1, 2))))
        src2 = self.dropout2(self.conv2(src2).transpose(1, 2))

        src = self.norm2(src + src2)
        return src


class Encoder(nn.Module):

    def __init__(
        self,
        encoder_layer,
        conv_layer=None,
        num_layers=2,
        norm_layer=None,
    ):
        super(Encoder, self).__init__()
        self.attn_layers = _get_clones(encoder_layer, num_layers)
        self.conv_layers = _get_clones(conv_layer, num_layers -
                                       1) if conv_layer is not None else None
        self.norm = norm_layer

    def forward(self, src, attn_mask=None) -> torch.Tensor:
        # x [B, L, D]
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers,
                                              self.conv_layers):
                output = attn_layer(src, attn_mask=attn_mask)
                output = conv_layer(output)
            output = self.attn_layers[-1](output)
        else:
            for attn_layer in self.attn_layers:
                output = attn_layer(output, attn_mask=attn_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DecoderLayer(nn.Module):

    def __init__(self,
                 self_attn_layer,
                 cross_attn_layer,
                 d_model,
                 d_ffn=None,
                 dropout=0.1,
                 activation='relu'):
        super(DecoderLayer, self).__init__()
        d_ffn = d_ffn or 4 * d_model
        self.self_attention = self_attn_layer
        self.cross_attention = cross_attn_layer
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ffn, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ffn, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None) -> torch.Tensor:

        tgt2 = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attention(
            tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.dropout3(self.activation(self.conv1(tgt.transpose(1, 2))))
        tgt2 = self.dropout4(self.conv2(tgt2).transpose(1, 2))
        tgt = self.norm3(tgt + tgt2)

        return tgt


class Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm_layer

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None) -> torch.Tensor:

        for layer in self.layers:
            output = layer(
                tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
