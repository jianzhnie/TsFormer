<!--
 * @Author: jianzhnie
 * @Date: 2022-01-20 17:31:27
 * @LastEditTime: 2022-01-26 10:54:25
 * @LastEditors: jianzhnie
 * @Description:
 *
-->
# TsFormer
TsFormer is a toolbox that implement transformer models on Time series data


## Todo

1. data preprocess
- gefcom2014
- uci
- ETT

2. dataloader

- dataloaders

3. models

- rnn
- lstm
- GRU
- ESN
- CNN
- TCN
- transformer
- informer
- autoformer

4. TODO
- [Spacetimeformer](https://github.com/QData/spacetimeformer)
- [SCINet](https://github.com/cure-lab/SCINet)
- [Deep learning for time series forecasting](https://github.com/AIStream-Peelout/flow-forecast)
- [PyTorch Forecasting](https://github.com/jdb78/pytorch-forecasting)
- [tsai](https://github.com/timeseriesAI/tsai)
- [flow-forecast](https://github.com/AIStream-Peelout/flow-forecast)

5. train and evaluate

<p align="center">
<img src="./docs/results.png" height = "550" alt="" align=left />
</p>

## 6. Custom  Informer

```sh
python -u run_autoformer.py \
        --is_training 1 \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL \
        --model informer \
        --data custom \
        --features S \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --embed fixed \
        --des 'Exp' \
        --itr 1


mse:0.2755982279777527, mae:0.3857262134552002,rmse:0.524974524974823, mape:1.9572646617889404, mspe:238.20448303222656
```


```sh
python -u run_autoformer.py \
        --is_training 1 \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL \
        --model informer \
        --data custom \
        --features S \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --embed timeF \
        --des 'Exp' \
        --itr 1

mse:0.22287048399448395, mae:0.3356129825115204,rmse:0.4720916152000427, mape:1.6913783550262451, mspe:260.3700866699219
```

### Transformer Results

```sh
python -u run_autoformer.py \
        --is_training 1 \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL \
        --model transformer \
        --data custom \
        --features S \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --embed timeF \
        --des 'Exp' \
        --itr


mse:0.284598171710968, mae:0.38772597908973694,rmse:0.5334774255752563, mape:2.1156060695648193, mspe:381.4866943359375
```


## AutoFormer

```sh
python -u run_autoformer.py \
        --is_training 1 \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL \
        --model autoformer \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --des 'Exp' \
        --itr 1

mse:0.2043592780828476, mae:0.3170555830001831,rmse:0.45206114649772644, mape:3.2521157264709473, mspe:414847.125
```
