# add --individual for P-sLSTM


if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=104

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/illness/ \
  --data_path national_illness.csv \
  --model_id Illness_$seq_len'_'24 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 24 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0005 \
  --train_epochs 20 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 32 --group_norm_weight True \
  --dropout 0.1 >logs/LongForecasting/P_sLSTM_illness_$seq_len'_'24.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/illness/ \
  --data_path national_illness.csv \
  --model_id Illness_$seq_len'_'36 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 36 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0005 \
  --train_epochs 20 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 32 --group_norm_weight True \
  --dropout 0.1 >logs/LongForecasting/P_sLSTM_illness_$seq_len'_'36.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/illness/ \
  --data_path national_illness.csv \
  --model_id Illness_$seq_len'_'48 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 48 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0005 \
  --train_epochs 20 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 32 --group_norm_weight True \
  --dropout 0.1 >logs/LongForecasting/P_sLSTM_illness_$seq_len'_'48.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/illness/ \
  --data_path national_illness.csv \
  --model_id Illness_$seq_len'_'60 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 60 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0005 \
  --train_epochs 20 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 32 --group_norm_weight True \
  --dropout 0.1 >logs/LongForecasting/P_sLSTM_illness_$seq_len'_'60.log


