export CUDA_VISIBLE_DEVICES="0,1,2,3"
model=Linear
methods_m='Linear'

percent=100
patience=10
tag_file=main.py

inp_len_m=512
pre_lens_m="96 192 336 720"
filename_m=ETTm_simple.txt
lr=0.0001
gpu_loc=0
itt=1
for pred_len in $pre_lens_m;
do
for method in $methods_m;
do
python $tag_file \
    --root_path ../datasets/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id 'ETTm1_'$inp_len_m'_'$pred_len'_simple_'$method \
    --data ett_m \
    --seq_len $inp_len_m \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 1024 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --percent $percent \
    --gpu_loc $gpu_loc \
    --patience $patience \
    --itr $itt \
    --method $method \
    --model $model \
    --cos 1 \
    --save_file_name $filename_m
done
done

# ETTm2
for pred_len in $pre_lens_m;
do
for method in $methods_m;
do
python $tag_file \
    --root_path ../datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id 'ETTm2_'$inp_len_m'_'$pred_len'_simple_'$method \
    --data ett_m \
    --seq_len $inp_len_m \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 1024 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --percent $percent \
    --method $method \
    --gpu_loc $gpu_loc \
    --itr $itt \
    --patience $patience \
    --model $model \
    --cos 1 \
    --save_file_name $filename_m
done
done

