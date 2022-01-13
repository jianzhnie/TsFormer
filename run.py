import argparse
import random

import numpy as np
import torch

from tsformer.exp_main import Exp_Main

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(
    description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument(
    '--is_training', type=int, required=True, default=1, help='status')
parser.add_argument(
    '--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument(
    '--model',
    type=str,
    required=True,
    default='Autoformer',
    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument(
    '--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument(
    '--root_path',
    type=str,
    default='./data/ETT/',
    help='root path of the data file')
parser.add_argument(
    '--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument(
    '--features',
    type=str,
    default='M',
    help='forecasting task, options:[M, S, MS];')
parser.add_argument(
    '--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument(
    '--freq', type=str, default='h', help='freq for time features encoding')
parser.add_argument(
    '--checkpoints',
    type=str,
    default='./checkpoints/',
    help='location of model checkpoints')

# forecasting task
parser.add_argument(
    '--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument(
    '--label_len', type=int, default=48, help='start token length')
parser.add_argument(
    '--pred_len', type=int, default=96, help='prediction sequence length')

# model define
parser.add_argument(
    '--input_size', type=int, default=1, help='encoder input size')
parser.add_argument('--hidden_size', type=int, default=7, help='hidden size')
parser.add_argument('--num_layers', type=int, default=7, help='num layers')
parser.add_argument('--output_size', type=int, default=512, help='output size')
parser.add_argument(
    '--embed',
    type=str,
    default='timeF',
    help='time features encoding, options:[timeF, fixed, learned]')

# optimization
parser.add_argument(
    '--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument(
    '--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='batch size of train input data')
parser.add_argument(
    '--patience', type=int, default=3, help='early stopping patience')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.0001,
    help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument(
    '--do_predict',
    action='store_true',
    help='whether to predict unseen future data')
parser.add_argument(
    '--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument(
    '--use_amp',
    action='store_true',
    help='use automatic mixed precision training',
    default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument(
    '--use_multi_gpu',
    action='store_true',
    help='use multiple gpus',
    default=False)
parser.add_argument(
    '--devices',
    type=str,
    default='0,1,2,3',
    help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}'.format(
            args.model_id, args.model, args.data, args.features, args.seq_len,
            args.label_len, args.pred_len, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(
            setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(
            setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.
                  format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model_id, args.model, args.data, args.features, args.seq_len,
        args.label_len, args.pred_len, args.d_model, args.n_heads,
        args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
        args.distil, args.des, ii)

    exp = Exp(args)  # set experiments
    print(
        '>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
