#!/bin/bash
dataset1="texas"
dataset2="photo"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset2=('Amherst41')

# nembed_lst=(32 16)
beta_lst=(1)

gpu=3
lr=0.001
nfactor=3
weight_decay=5e-4
nfeat=128
nhidden=(1024)
nembed=32
epochs=2000
layer=1
temperature=1
loss_weight=20
run=10

mini=(0)
sub='DE'
mi=1
for hid in "${nhidden[@]}"; do
		python ./main_avgae.py --dataset $dataset1 \
            --sub_dataset $sub_dataset2\
            --miniid $mi\
            --nhidden $hid \
            --lr $lr \
            --nfactor $nfactor \
            --weight_decay $weight_decay \
            --nfeat $nfeat \
            --nembed $nembed \
            --epochs $epochs \
            --layer $layer \
            --temperature $temperature \
            --loss_weight $loss_weight \
            --run $run    \
            --gpu $gpu   
done 