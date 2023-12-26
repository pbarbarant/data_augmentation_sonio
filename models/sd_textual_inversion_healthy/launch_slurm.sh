#!/bin/bash
srun --job-name=accelerate_multi-node \
    --partition=gpu \
    --gres=gpu:1 \
    --time=70:00:00 \
    bash train.sh
