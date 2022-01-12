'''
Author: jianzhnie
Date: 2022-01-12 11:49:33
LastEditTime: 2022-01-12 12:18:06
LastEditors: jianzhnie
Description:

'''
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tsformer.datasets.rnn_dataloader import RNNDataset
from tsformer.models.rnn_model import LSTM

if __name__ == '__main__':
    embedding_dim = 128
    hidden_dim = 256
    batch_size = 32
    num_epoch = 5
    root_path = 'data/GEFCom2014/Load'
    target = 'LOAD'

    # 加载数据
    train_dataset = RNNDataset(
        root_path=root_path, flag='train', target=target)
    test_dataset = RNNDataset(root_path=root_path, flag='val', target=target)
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM()
    model.to(device)

    # 训练过程
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_data_loader, desc=f'Training Epoch {epoch}'):
            inputs, targets = [x.to(device) for x in batch]
            print(inputs.shape)
            print(targets.shape)
            log_probs = model(inputs)
            loss = mse_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss: {total_loss:.2f}')
