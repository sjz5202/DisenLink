#!/bin/bash
dataset1="twitch-e"
dataset2="Reed98"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset2=('Amherst41' 'Johns Hopkins55')
dataset=('fb100')
# nembed_lst=(32 16)
beta_lst=(1)

gpu=0
lr=0.001
weight_decay=5e-4
nembed=32
epochs=2000
run=2
m=5
head=5
nhidden=256

for sub in "${dataset[@]}"; do
		python ./main_gat.py --dataset $sub \
            --sub_dataset $dataset2\
            --lr $lr \
            --weight_decay $weight_decay \
            --head $head    \
            --nhidden $nhidden  \
            --nembed $nembed \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m 
done 