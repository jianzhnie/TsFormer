'''
Author: jianzhnie
Date: 2022-01-25 10:43:34
LastEditTime: 2022-01-26 10:34:04
LastEditors: jianzhnie
Description:

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_informer import Encoder, _get_clones
from .layers.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.embed import DataEmbedding_wo_pos


class AutoFormer(nn.Module):

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
                 moving_avg=25,
                 freq='h',
                 activation='gelu'):

        super(AutoFormer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.c_out = c_out

        # Decomp
        kernel_size = moving_avg
        self.decomp = SeriesDecomp(kernel_size)

        # Encoding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq,
                                                  dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq,
                                                  dropout)
        # Attention
        enc_attn_fun = AutoCorrelation(
            False, factor, attention_dropout=dropout)

        # Encoder
        encoder_norm = SeasonalLayerNorm(d_model)
        enc_attn_layer = AutoCorrelationLayer(enc_attn_fun, d_model, n_heads)
        encoder_layer = EncoderLayer(
            attention_layer=enc_attn_layer,
            d_model=d_model,
            moving_avg=moving_avg,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation)

        self.encoder = Encoder(
            encoder_layer=encoder_layer,
            conv_layer=None,
            num_layers=e_layers,
            norm_layer=encoder_norm)

        # Decoder
        dec_attn_fun1 = AutoCorrelation(
            True, factor, attention_dropout=dropout)
        dec_attn_fun2 = AutoCorrelation(
            False, factor, attention_dropout=dropout)
        dec_attn_layer1 = AutoCorrelationLayer(dec_attn_fun1, d_model, n_heads)
        dec_attn_layer2 = AutoCorrelationLayer(dec_attn_fun2, d_model, n_heads)
        decoder_layer = DecoderLayer(
            self_attn_layer=dec_attn_layer1,
            cross_attn_layer=dec_attn_layer2,
            d_model=d_model,
            c_out=c_out,
            d_ffn=d_ffn,
            moving_avg=moving_avg,
            dropout=dropout,
            activation=activation)
        decoder_norm = SeasonalLayerNorm(d_model)
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
        # decomp init
        mean = torch.mean(
            x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]],
                            device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean],
                               dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            tgt_mask=dec_self_mask,
            memory_mask=dec_enc_mask,
            trend=trend_init)

        seasonal_part = self.projection(seasonal_part)
        # final
        dec_out = trend_part + seasonal_part

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class SeasonalLayerNorm(nn.Module):
    """Special designed layernorm for the seasonal part."""

    def __init__(self, channels):
        super(SeasonalLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class MovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """Autoformer encoder layer with the progressive decomposition
    architecture."""

    def __init__(self,
                 attention_layer,
                 d_model,
                 d_ffn=None,
                 moving_avg=25,
                 dropout=0.1,
                 activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ffn = d_ffn or 4 * d_model
        self.attention = attention_layer
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ffn, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(
            in_channels=d_ffn, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, src, attn_mask=None):
        """src: batch_size * seq_length * hidden_size
        """
        src2 = self.attention(src, src, src, attn_mask=attn_mask)[0]
        src = src + self.dropout1(src2)
        src, _ = self.decomp1(src)

        src2 = self.dropout2(self.activation(self.conv1(src.transpose(1, 2))))
        src2 = self.dropout3(self.conv2(src2).transpose(1, 2))
        src, _ = self.decomp2(src + src2)
        return src


class DecoderLayer(nn.Module):
    """Autoformer decoder layer with the progressive decomposition
    architecture."""

    def __init__(self,
                 self_attn_layer,
                 cross_attn_layer,
                 d_model,
                 c_out,
                 d_ffn=None,
                 moving_avg=25,
                 dropout=0.1,
                 activation='relu'):
        super(DecoderLayer, self).__init__()
        d_ffn = d_ffn or 4 * d_model
        self.self_attention = self_attn_layer
        self.cross_attention = cross_attn_layer
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ffn, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(
            in_channels=d_ffn, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='circular',
            bias=False)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt, trend1 = self.decomp1(tgt)

        tgt2 = self.cross_attention(
            tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt, trend2 = self.decomp2(tgt)

        tgt2 = self.dropout3(self.activation(self.conv1(tgt.transpose(1, 2))))
        tgt2 = self.dropout4(self.conv2(tgt2).transpose(1, 2))
        tgt, trend3 = self.decomp3(tgt + tgt2)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1))
        residual_trend = residual_trend.transpose(1, 2)
        return tgt, residual_trend


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
                memory_mask=None,
                trend=None) -> torch.Tensor:

        for layer in self.layers:
            tgt, residual_trend = layer(
                tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

            trend = trend + residual_trend

        if self.norm is not None:
            tgt = self.norm(tgt)
        return tgt, trend
