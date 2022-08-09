#!/bin/bash
dataset1="twitch-e"
dataset2="fb100"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset2=('Amherst41' 'Johns Hopkins55')
dataset=('wisconsin')
# nembed_lst=(32 16)
beta_lst=(1)
mini=(0 1 2 3 4 5 6 7 8 9)
gpu=1
lr=0.001
weight_decay=5e-4
nembed=32
epochs=2000
run=10
m=5
head=5
nhidden=(256)
mi=1
for hid in "${nhidden[@]}"; do
		python ./main_gat.py --dataset $dataset \
            --sub_dataset $dataset2\
            --lr $lr \
            --weight_decay $weight_decay \
            --head $head    \
            --nhidden $hid  \
            --nembed $nembed \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m \
            --miniid $mi
done 