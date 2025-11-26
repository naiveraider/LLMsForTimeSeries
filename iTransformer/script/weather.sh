export CUDA_VISIBLE_DEVICES="0,1,2,3"

seq_len=96

model=iTransformer
methods_h='iTransformer'

gpu_loc=0
percent=100
filename=weather

pre_lens_h='96 192 336 720'
for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
python run.py \
    --is_training 1 \
    --root_path ../datasets/weather/ \
    --data_path weather.csv \
    --model_id 'weather_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --model $model \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --e_layers 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --gpu $gpu_loc
done
done

