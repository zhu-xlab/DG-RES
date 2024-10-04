#!/bin/bash

device=4
data='PACS'
network='resnet18'

for t in `seq 0 4`
do
  for domain in `seq 0 3`
  do
    python train_domain.py \
    --target $domain \
    --device $device \
    --network $network \
    --time $t \
    --batch_size 64 \
    --data $data \
    --data_root "./data/DataSets/" \
    --result_path "./data/save/models/" \
    --epochs 50 \
    --learning_rate 0.002
  done
done
