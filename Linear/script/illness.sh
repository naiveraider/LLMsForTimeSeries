export CUDA_VISIBLE_DEVICES="0,1,2"

seq_len=104
percent=100
gpu_loc=0 
model=Linear
methods_h='Linear'

tag_file=main.py
 

filename=Illness_simple.txt 
pre_lens_h="24 36 48 60"
for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
lr=0.0001
bs=8
python $tag_file\
    --root_path ../datasets/illness/ \
    --data_path national_illness.csv \
    --model_id 'Illness_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --method $method \
    --pred_len $pred_len \
    --batch_size $bs \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --all 1 \
    --percent $percent \
    --itr 1 \
    --model $model \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done

