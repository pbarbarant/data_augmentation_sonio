#!/bin/bash
srun --job-name=accelerate_multi-node \
    --partition=gpu \
    --gres=gpu:ampere:1 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=32 \
    --time=70:00:00 \
    bash train.sh
