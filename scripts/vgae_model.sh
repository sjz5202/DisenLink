#!/bin/bash
dataset1="twitch-e"
dataset2="photo"
dataset3="year"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset2=('JohnsHopkins55')
mini=(0)
# nembed_lst=(32 16)
beta_lst=(1)
gpu=4
lr=0.001
weight_decay=5e-4
nembed=32
epochs=2000
run=10
m=5
nhidden=(128)
ind=(0)
id=1
for hid in "${nhidden[@]}"; do
		python ./main_vgae.py --dataset $dataset2 \
            --lr $lr \
            --sub_dataset $sub_dataset2\
            --weight_decay $weight_decay \
            --nembed $nembed \
            --nhidden $hid \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m \
            --miniid $id
done 