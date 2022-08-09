#!/bin/bash
dataset="twitch-e"
sub_dataset="ES"
# "DE", "ENGB", "ES", "FR", "PTBR", "RU", "TW"
# 'Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'

# nembed_lst=(32 16)
beta_lst=(1)
dataset=('crocodile' 'chameleon' 'squirrel')
gpu=1
lr=0.001
nfactor=3
weight_decay=5e-4
nfeat=128
nhidden=128
nembed=32
epochs=2000
layer=1
temperature=1
loss_weight=20
run=2


for data in "${dataset[@]}"; do
		python ./GIC_main.py --dataset $data \
            --sub_dataset $sub_dataset\
            --nhidden $nhidden \
            --lr $lr \
            --beta $beta \
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
            --gpu $gpu   
done 