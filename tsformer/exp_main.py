import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from tsformer.datasets.data_factory import data_provider
from tsformer.exp_basic import Exp_Basic
from tsformer.models.rnn_model import CNN, GRU, LSTM, RNN, AttentionalLSTM
from tsformer.models.transformer import Transformer
from tsformer.utils.metrics import metric
from tsformer.utils.tools import EarlyStopping, visual

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):

        input_size = self.args.input_size
        hidden_size = self.args.hidden_size
        num_layers = self.args.num_layers
        output_size = self.args.output_size

        if self.args.model == 'cnn':
            model = CNN(input_size, hidden_size, output_size)
        if self.args.model == 'rnn':
            model = RNN(input_size, hidden_size, num_layers, output_size)
        elif self.args.model == 'lstm':
            model = LSTM(input_size, hidden_size, num_layers, output_size)
        elif self.args.model == 'gru':
            model = GRU(input_size, hidden_size, num_layers, output_size)
        elif self.args.model == 'attlstm':
            model = AttentionalLSTM(
                input_size=input_size,
                qkv=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size)

        elif self.args.model == 'transformer':

            model = Transformer(
                input_features=input_size,
                input_seq_len=96,
                hidden_dim=768,
                output_seq_len=output_size,
                dim_feedforward=512,
                num_head=12,
                num_layers=2,
                dropout=0.1,
            )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_lr_scheduler(self, epochs):
        optimizer = self._select_optimizer()
        scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
        return scheduler

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                outputs = outputs.unsqueeze(-1)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.results_dir, 'checkpoints', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        lr_scheduler = self._get_lr_scheduler(self.args.train_epochs)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                outputs = outputs.unsqueeze(-1)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print('\titers: {0}, epoch: {1} | loss: {2:.7f}'.format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(
                        speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print('Epoch: {} cost time: {}'.format(epoch + 1,
                                                   time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            lr = lr_scheduler.get_last_lr()

            print(
                'Epoch: {0}, Lr:{1} |  Steps: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}'
                .format(epoch + 1, lr, train_steps, train_loss, vali_loss,
                        test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print('Early stopping')
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            lr_scheduler.step()

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.args.results_dir, 'checkpoints', setting,
                                 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = os.path.join(self.args.results_dir, 'test_results',
                                   setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                outputs = outputs.unsqueeze(-1)
                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]),
                                        axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]),
                                        axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        # result save
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        folder_path = os.path.join(self.args.results_dir, 'results', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{},rmse:{}, mape:{}, mspe:{}'.format(
            mse, mae, rmse, mape, mspe))
        test_result_file = os.path.join(folder_path, 'result.txt')
        with open(test_result_file, 'w+') as f:
            f.write(setting + '  \n')
            f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe{}'.format(
                mse, mae, rmse, mape, mspe))
            f.write('\n')
            f.write('\n')
            f.close()

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.results_dir, 'checkpoints', setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        df = pd.DataFrame(preds)
        # result save
        folder_path = os.path.join(self.args.results_dir + setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        df.to_csv(folder_path + 'real_prediction.csv', index=False)

        return
