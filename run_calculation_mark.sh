TIME=$(date "+%m-%d-%H-%M")
DATASET=calculation_mark
TEMPLATE=vanilla

# wandb
export WANDB_PROJECT=xukp20-$DATASET

# set HF_HOME env
# export HF_HOME=/lustre/cache/huggingface
OUTPUT_DIR=/cephfs/xukangping/root/models/$WANDB_PROJECT/$TEMPLATE-$DATASET-$TIME
# OUTPUT_DIR=~/models/llama-tuned/codellama-34b-$DATASET-$TIME

# mistral 7b
# MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/5e9c98b96d071dce59368012254c55b0ec6f8658"

# gemma 2b
# MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--google--gemma-2b/snapshots/9d067f00def958594aaa16b39a65b07d69ca655b"

# phi-2
# MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--microsoft--phi-2/snapshots/710686f446f02286c858c11f052acb87c306ddd2"

# llama-3 
MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b"

VAL_SIZE=0.01
NUM_GPUS=8
BATCH_SIZE=2
# LR=5e-5
LR=1e-5 # for meta_math
EPOCHS=5
# MAX_LEN=2048
MAX_LEN=4096
SAVE_STEPS=50

# load custom model
# export CUSTOM_MODEL_PATH="/cephfs/xukangping/code/LLaMA-Efficient-Tuning/load_my_model.py"

# accelerate launch src/train_bash.py \
# deepspeed --hostfile hostfile.txt src/train_bash.py \
deepspeed --num_gpus=8 --master_port=9901 src/train_bash.py \
    --deepspeed "/cephfs/xukangping/code/LLaMA-Efficient-Tuning/ds_config_zero1.json" \
    --stage sft \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --do_train True \
    --overwrite_cache False \
    --finetuning_type full \
    --template $TEMPLATE \
    --dataset_dir data \
    --dataset $DATASET \
    --cutoff_len $MAX_LEN \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --max_samples 10000000 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps $SAVE_STEPS \
    --warmup_steps 0 \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --plot_loss True \
    --val_size $VAL_SIZE \
    --evaluation_strategy steps \
    --eval_steps 5 \
    --report_to wandb \
    --preprocessing_num_workers 32 \
    --flash_attn \
    # --load_best_model_at_end True