import math

import torch
import torch.nn as nn


class DNN(nn.Module):
    """Deep Neural Network."""

    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x = x.squeeze(dim=2)
        out = self.main(x)
        return out


class CNN(nn.Module):
    """Convolutional Neural Networks."""

    def __init__(self, input_size, hidden_dim, output_size):
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size, out_channels=hidden_dim,
                kernel_size=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(hidden_dim, 10), nn.Linear(10, output_size))

    def forward(self, x):
        out = self.main(x)
        return out


class RNN(nn.Module):
    """Vanilla RNN."""

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        hidden = hidden.view(-1, self.hidden_size)
        # torch.Size([batch_size, 128]) ==> torch.Size([32, output_size])
        out = self.dropout(hidden)
        out = self.fc(out)
        return out

    def forward_(self, x):
        # forward pass through rnn layer
        # shape of rnn_out:  [batch_size, seq_length, hidden_dim]
        # shape of self.hidden: (a, b) where a and b both have shape:  (num_layers, batch_size, hidden_dim)
        rnn_out, hidden = self.rnn(x)
        # Only take output from the final timestep
        # Can pass on the entirety  rnn_out to the next layer if it is a seq2seq prediction
        rnn_out = rnn_out[:, -1, :]
        rnn_out = self.fc(rnn_out)

        return rnn_out


class LSTM(nn.Module):
    """Long Short Term Memory."""

    def __init__(self,
                 input_size=1,
                 hidden_size=128,
                 num_layers=1,
                 output_size=1):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device)
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device)
        return (h_0, c_0)

    def forward_(self, x):
        # forward pass through LSTM layer
        # shape of lstm_out:  [batch_size, seq_length, hidden_dim]
        # shape of self.hidden: (a, b) where a and b both have shape:  (num_layers, batch_size, hidden_dim)
        lstm_out, (h_out, c_out) = self.lstm(x)
        # Only take output from the final timestep
        # Can pass on the entirety  lstm_out to the next layer if it is a seq2seq prediction
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.fc(lstm_out)

        return lstm_out

    def forward(self, x):
        # forward pass through LSTM layer
        # shape of lstm_out:  [batch_size, seq_length, hidden_dim]
        # shape of self.hidden: (a, b) where a and b both have shape:  (num_layers, batch_size, hidden_dim)
        # batch_size = x.shape[0]
        # device = x.device
        # h_0, c_0 = self.init_hidden(batch_size, device)
        lstm_out, (h_out, c_out) = self.lstm(x)
        # torch.Size([1, 32, 128]) ==> torch.Size([32, 128])
        h_out = h_out.view(-1, self.hidden_size)
        # torch.Size([32, 128]) ==> torch.Size([32, output_size])
        h_out = self.fc(h_out)
        return h_out


class GRU(nn.Module):
    """Gat e Recurrent Unit."""

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, hidden = self.gru(x)
        # torch.Size([1, 32, 128]) ==> torch.Size([32, 128])
        hidden = hidden.view(-1, self.hidden_size)
        # torch.Size([32, 128]) ==> torch.Size([32, output_size])
        out = self.fc(hidden)
        return out


class AttentionalLSTM(nn.Module):
    """LSTM with Attention."""

    def __init__(self,
                 input_size,
                 qkv,
                 hidden_size,
                 num_layers,
                 output_size,
                 bidirectional=False):
        super(AttentionalLSTM, self).__init__()

        self.input_size = input_size
        self.qkv = qkv
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.query = nn.Linear(input_size, qkv)
        self.key = nn.Linear(input_size, qkv)
        self.value = nn.Linear(input_size, qkv)

        self.attn = nn.Linear(qkv, input_size)
        self.scale = math.sqrt(qkv)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)

        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        Q, K, V = self.query(x), self.key(x), self.value(x)

        dot_product = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale
        scores = torch.softmax(dot_product, dim=-1)
        scaled_x = torch.matmul(scores, V) + x

        out = self.attn(scaled_x) + x
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        out = self.fc(out)

        return out
