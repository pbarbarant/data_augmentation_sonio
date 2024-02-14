#!/bin/bash
srun --job-name=accelerate_multi-node \
    --partition=gpu \
    --gres=gpu:volta \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=32 \
    --time=70:00:00 \
    bash models/sd_lora/train.sh
