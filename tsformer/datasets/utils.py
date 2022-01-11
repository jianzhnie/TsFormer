'''
Author: jianzhnie
Date: 2022-01-11 17:05:34
LastEditTime: 2022-01-11 17:09:06
LastEditors: jianzhnie
Description:

'''


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
