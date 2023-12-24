export MODEL_NAME="stabilityai/sd-turbo"
export DATA_DIR="/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/processed_by_modality"

python textual_inversion.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATA_DIR \
    --learnable_property="object" \
    --placeholder_token="<cardiac-healthy>" \
    --initializer_token="heart" \
    --resolution=512 \
    --train_batch_size=1 \
    --max_train_steps=50000 \
    --num_vectors=8 \
    --learning_rate=5.0e-03 \
    --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --save_steps=10000 \
    --output_dir="textual_inversion_output" \
    # --validation_prompt="A <cardiac-healthy> train" \
    # --num_validation_images=4 \
    # --validation_steps=100
