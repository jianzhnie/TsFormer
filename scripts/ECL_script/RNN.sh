export CUDA_VISIBLE_DEVICES=1
###
 # @Author: jianzhnie
 # @Date: 2022-01-13 15:20:03
 # @LastEditTime: 2022-01-14 10:46:11
 # @LastEditors: jianzhnie
 # @Description:
 #
###

python -u run.py \
      --is_training 1 \
      --root_path data/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_96_96 \
      --model rnn \
      --data RNNData \
      --features MS \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --input_size 321 \
      --hidden_size 256 \
      --num_layers 1 \
      --output_size 96 \
      --des 'Exp' \
      --itr 1
