export MODEL_NAME="stabilityai/sd-turbo"
# export MODEL_PATH="/data/parietal/store3/work/pbarbara/data_augmentation_sonio/models/sd_lora/sd_lora_output/pytorch_lora_weights.safetensors"
export OUTPUT_DIR="/data/parietal/store3/work/pbarbara/data_augmentation_sonio/models/sd_lora/sd_lora_output"
export DATASET_NAME="/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/processed_by_modality/patho"
accelerate launch --gpu_ids 0 models/sd_lora/train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATASET_NAME \
    --dataloader_num_workers=8 \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=20000 \
    --learning_rate=1e-04 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --output_dir=${OUTPUT_DIR} \
    --checkpointing_steps=1000 \
    --resume_from_checkpoint latest \
    --image_column="image" \
    --caption_column="caption" \
    --report_to="wandb"
