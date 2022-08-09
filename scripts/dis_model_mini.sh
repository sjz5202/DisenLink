#!/bin/bash
dataset='year'
mini=(1 2 3 4 5 6 7 8 9)
gpu=6
lr=0.001
nfactor=5
weight_decay=5e-4
nfeat=128
nhidden=256
nembed=32
epochs=2000
layer=1
temperature=1
run=5
m=5
#beta_list=(0.7 0.8 0.9 0.7 0.5 0.7)
beta_list=(0.7 0.8 0.9 0.7 0.5 0.7 0.6 0.8 0.6 0.7)
#ind=(0 1 2 3 4 5 6 7 8)
ind=(0 1 2 3 4 5 6 7 8 9)
for id in "${ind[@]}"; do
            python ./main_disentangled.py --dataset $dataset \
            --miniid $id\
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
            --run $run    \
            --gpu $gpu  \
            --m $m
done