'''
Author: jianzhnie
Date: 2022-01-11 17:05:34
LastEditTime: 2022-01-14 17:19:44
LastEditors: jianzhnie
Description:

'''

import numpy as np


# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, seq_len):
    inout_seq = []
    L = len(input_data)
    for i in range(L - seq_len):
        train_seq = input_data[i:i + seq_len]
        train_label = input_data[(i + seq_len):(i + seq_len + 1)]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def split_sequence_uni_step(sequence, n_steps):
    """Rolling Window Function for Uni-step."""
    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def split_sequence_multi_step(sequence, n_steps_in, n_steps_out):
    """Rolling Window Function for Multi-step."""
    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        if out_end_ix > len(sequence):
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)[:, :, 0]
