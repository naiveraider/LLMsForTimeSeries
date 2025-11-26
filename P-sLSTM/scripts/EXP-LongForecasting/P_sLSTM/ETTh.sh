# add --individual for P-sLSTM

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model P_sLSTM \
  --data ett_h \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 32 --learning_rate 0.00006 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 32 --group_norm_weight True \
  --dropout 0.1 >logs/LongForecasting/P_sLSTM_ETTh1_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'192 \
  --model P_sLSTM \
  --data ett_h \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 4 --conv1d_kernel_size 4 --group_norm_weight True \
  --dropout 0 >logs/LongForecasting/P_sLSTM_ETTh1_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'336 \
  --model P_sLSTM \
  --data ett_h \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 4 --conv1d_kernel_size 4 --group_norm_weight True \
  --dropout 0 >logs/LongForecasting/P_sLSTM_ETTh1_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'720 \
  --model P_sLSTM \
  --data ett_h \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.00006 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 4 --conv1d_kernel_size 2 --group_norm_weight True \
  --dropout 0 >logs/LongForecasting/P_sLSTM_ETTh1_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'96 \
  --model P_sLSTM \
  --data ett_h \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 32 --learning_rate 0.00006 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 32 --group_norm_weight True \
  --dropout 0.1 >logs/LongForecasting/P_sLSTM_ETTh2_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'192 \
  --model P_sLSTM \
  --data ett_h \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 4 --conv1d_kernel_size 4 --group_norm_weight True \
  --dropout 0 >logs/LongForecasting/P_sLSTM_ETTh2_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'336 \
  --model P_sLSTM \
  --data ett_h \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 4 --conv1d_kernel_size 4 --group_norm_weight True \
  --dropout 0 >logs/LongForecasting/P_sLSTM_ETTh2_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'720 \
  --model P_sLSTM \
  --data ett_h \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.00006 \
  --patch_size 6 --stride 6 \
  --num_blocks 1 \
  --channel 7 --embedding_dim 100 --num_heads 4 --conv1d_kernel_size 2 --group_norm_weight True \
  --dropout 0 >logs/LongForecasting/P_sLSTM_ETTh2_$seq_len'_'720.log


