export CUDA_VISIBLE_DEVICES="0,1,2"

seq_len=104

model=iTransformer
methods_h='iTransformer'

gpu_loc=0
percent=100
filename=illness

pre_lens_h="24 36 48 60"
for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
python run.py \
    --is_training 1 \
    --root_path ../datasets/illness/ \
    --data_path national_illness.csv \
    --model_id 'Illness_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --model $model \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 256 \
    --itr 1 \
    --gpu $gpu_loc
done
done

