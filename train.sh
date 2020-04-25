#!bin/bash

for iters in 1 2 3 4;
do
    CUDA_VISIBLE_DEVICES=2 python3 train.py --iters=${iters}
done

