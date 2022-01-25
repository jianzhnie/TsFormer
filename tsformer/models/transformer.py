'''
Author: jianzhnie
Date: 2022-01-17 16:36:57
LastEditTime: 2022-01-25 10:05:56
LastEditors: jianzhnie
Description:

'''

import torch
import torch.nn as nn

from .custom_informer import Decoder, DecoderLayer, Encoder, EncoderLayer
from .layers.attn import AttentionLayer, FullAttention
from .layers.embed import DataEmbedding


class Transformer(nn.Module):

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
        super(Transformer, self).__init__()
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
        enc_full_attn = FullAttention(False, factor, attention_dropout=dropout)

        # Encoder
        encoder_norm = nn.LayerNorm(d_model)
        enc_attn_layer = AttentionLayer(enc_full_attn, d_model, n_heads)
        encoder_layer = EncoderLayer(enc_attn_layer, d_model, d_ffn, dropout,
                                     activation)

        self.encoder = Encoder(
            encoder_layer=encoder_layer,
            conv_layer=None,
            num_layers=e_layers,
            norm_layer=encoder_norm)

        # Decoder
        dec_prob_attn = FullAttention(True, factor, attention_dropout=dropout)
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
