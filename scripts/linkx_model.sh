#!/bin/bash
dataset='wisconsin'
mini=(0)
gpu=4
lr=0.001
nfactor=5
weight_decay=5e-4
nfeat=128
nhidden=(128)
nembed=32
epochs=2000
layer=1
temperature=1
run=10
m=5
id=1
#ind=(0 1 2 3 4 5 6 7 8)
for hid in "${nhidden[@]}"; do
            python ./main_linkx.py --dataset $dataset \
            --miniid $id\
            --lr $lr \
            --nfactor $nfactor \
            --weight_decay $weight_decay \
            --nfeat $nfeat \
            --nhidden $hid \
            --nembed $nembed \
            --epochs $epochs \
            --layer $layer \
            --temperature $temperature \
            --run $run    \
            --gpu $gpu  \
            --m $m
done