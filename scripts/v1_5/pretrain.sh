#!/bin/bash

lm=$1            # lmsys/vicuna-13b-v1.5
version=$2       # plain
data_path=$3     # ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json
image_folder=$4  # ./playground/data/LLaVA-Pretrain/images
output_dir=$5    # ./checkpoints/llava-v1.5-13b-pretrain

# this script assumes you're running with 4 40GB A100 GPUs
# (pre-training should take < 16 hours as it took ~3.5 w/ 8 80GB A100s)

deepspeed mllava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $lm \
    --version $version \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
