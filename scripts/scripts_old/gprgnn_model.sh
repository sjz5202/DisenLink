#!/bin/bash
dataset1="twitch-e"
dataset2="fb100"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
#sub_dataset1=("ENGB")
#sub_dataset2=('Amherst41' 'Johns Hopkins55')
sub_dataset2=('Amherst41')
sub='Reed98'
# nembed_lst=(32 16)
beta_lst=(1)

gpu=0
lr=0.001
weight_decay=5e-4
nembed=32
epochs=2000
run=5
m=5
ind=(0)
alpha_list=(0.5)
dataset=('crocodile' 'chameleon' 'squirrel')
for a in "${alpha_list[@]}"; do
        python ./main_gprgnn.py --dataset $dataset2 \
            --sub_dataset $sub\
            --lr $lr \
            --weight_decay $weight_decay \
            --nembed $nembed \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m \
            --alpha $a
done 
