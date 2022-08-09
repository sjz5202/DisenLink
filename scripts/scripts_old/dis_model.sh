#!/bin/bash
dataset1="twitch-e"
dataset2="fb100"
dataset3="squirrel"
#sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset1=('Johns Hopkins55')
#sub_dataset1=("ES" "FR" "RU" "PTBR" "TW")
#sub_dataset2=('Johns Hopkins55')
sub_dataset2=('Amherst41')
sub_dataset="ES"
gpu=0
lr=0.0001
nfactor=2
weight_decay=5e-4
nfeat=128
nhidden=256
nembed=32
epochs=2000
layer=1
temperature=1
loss_weight=20
run=1
m=5
#beta_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
beta_list=(0.7)
#ind=(0 1 2 3 4 5 6 7 8 9)
ind=(0)


for id in "${ind[@]}"; do
            python ./main_disentangled.py --dataset $dataset3 \
            --sub_dataset ${sub_dataset1[0]}\
            --nhidden $nhidden \
            --lr $lr \
            --beta ${beta_list[id]} \
            --nfactor $nfactor \
            --weight_decay $weight_decay \
            --nfeat $nfeat \
            --nhidden $nhidden \
            --nembed $nembed \
            --epochs $epochs \
            --layer $layer \
            --temperature $temperature \
            --loss_weight $loss_weight \
            --run $run    \
            --gpu $gpu  \
            --m $m
done