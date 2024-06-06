TIME=$(date "+%m-%d-%H-%M")
DATASET=metamathQA
TEMPLATE=alpaca

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

# phi-2 copyied 48
# MODEL_NAME_OR_PATH=/cephfs/xukangping/root/models/phi-2-48

# llama-3 
# MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b"

# uni-phi
# MODEL_NAME_OR_PATH="/cephfs/xukangping/root/models/uni-phi-2"

# uni-phi v2
# MODEL_NAME_OR_PATH="/cephfs/xukangping/root/models/uni-phi-2-0512"

# MODEL_NAME_OR_PATH="/cephfs/xukangping/root/models/uni-phi-2-0512-tcr"

# uni-loop-phi
# MODEL_NAME_OR_PATH="/cephfs/xukangping/root/models/uni-coloop-phi-0514"

# uni-loop-phi cov
MODEL_NAME_OR_PATH=/cephfs/xukangping/root/models/uni-coloop-phi-cov-0516

RESUME_FROM_CHECKPOINT=/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-05-16-23-57/checkpoint-9000
MODEL_NAME_OR_PATH=$RESUME_FROM_CHECKPOINT

VAL_SIZE=0.005
NUM_GPUS=8
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
# LR=5e-5
LR=1e-5 # for meta_math
EPOCHS=6
MAX_LEN=2048
SAVE_STEPS=3000

# for continue the gate training
# VAL_SIZE=0.005
# NUM_GPUS=8
# BATCH_SIZE=4
# # LR=5e-5
# LR=2e-6 # for meta_math
# EPOCHS=4
# MAX_LEN=2048
# SAVE_STEPS=3000

# load custom model
# export CUSTOM_MODEL_PATH="/cephfs/xukangping/code/LLaMA-Efficient-Tuning/load_my_model.py"
export FIX_PHI="0"

if [ $FIX_PHI == "1" ]; then
    OUTPUT_DIR=$OUTPUT_DIR"-fix"
fi
if [ ! -z $CUSTOM_MODEL_PATH ]; then
    OUTPUT_DIR=$OUTPUT_DIR"-custom"
fi

echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "FIX_PHI: $FIX_PHI"

export WANDB_EVAL_CALLBACK=1


# accelerate launch src/train_bash.py \
# deepspeed --hostfile hostfile.txt src/train_bash.py \
deepspeed --num_gpus=$NUM_GPUS --master_port=9901 src/train_bash.py \
    --deepspeed "/cephfs/xukangping/code/LLaMA-Efficient-Tuning/ds_config_zero1.json" \
    --stage sft \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
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
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps $SAVE_STEPS \
    --warmup_steps 0 \
    --lora_rank 8 \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --plot_loss True \
    --val_size $VAL_SIZE \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --report_to wandb \
    --preprocessing_num_workers 32 \
    --flash_attn
    # --load_best_model_at_end True