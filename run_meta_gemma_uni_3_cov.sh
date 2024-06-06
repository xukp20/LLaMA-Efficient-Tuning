TIME=$(date "+%m-%d-%H-%M")
DATASET=metamathQA
TEMPLATE=alpaca

# wandb
export WANDB_PROJECT=xukp20-$DATASET-gemma-e3

RUN_NAME=$1
if [ -z $RUN_NAME ]; then
    RUN_NAME_PAR=""
else
    RUN_NAME_PAR="--run_name $RUN_NAME"
fi

SEED=$2
if [ -z $SEED ]; then
    SEED=42
fi

# set HF_HOME env
# export HF_HOME=/lustre/cache/huggingface
OUTPUT_DIR=/cephfs/xukangping/root/models/$WANDB_PROJECT/$TEMPLATE-$DATASET-$TIME


# phi-2
if [ $RUN_NAME == "base" ]; then
    MODEL_NAME_OR_PATH=/cephfs/shared/hf_cache/hub/models--google--gemma-2b/snapshots/2ac59a5d7bf4e1425010f0d457dde7d146658953
fi

if [ $RUN_NAME == "cov_base_001_all" ]; then
    echo "error"
fi

if [ $RUN_NAME == "base_48" ]; then
    MODEL_NAME_OR_PATH=/cephfs/xukangping/root/models/gemma-2b-28
fi

if [ $RUN_NAME == "cov_base_48_001_all" ]; then
    echo "error"
fi

# uni-loop-phi cov
if [ $RUN_NAME == "cov_1_001_all" ]; then
    MODEL_NAME_OR_PATH=/cephfs/xukangping/root/models/gemma-2b-loop
fi

if [ $RUN_NAME == "gram_1_001_all" ]; then
    # error
    echo "error"
fi

if [ $RUN_NAME == "cov_2_001_all" ]; then
    echo "error"
fi

if [ $RUN_NAME == "gram_2_001_all" ]; then
    echo "error"
fi

if [ $RUN_NAME == "1_0_none" ]; then
    echo "error"
fi

if [ $RUN_NAME == "2_0_none" ]; then
    echo "error"
fi

if [ $RUN_NAME == "3_0_none" ]; then
    echo "error"
fi

if [ $RUN_NAME == "cov_2_001_28831" ]; then
    echo "error"
fi

VAL_SIZE=0.005
NUM_GPUS=8
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-5 # for meta_math
EPOCHS=3
MAX_LEN=2048
SAVE_STEPS=3000

# export WANDB_EVAL_CALLBACK=1

if [ -z $RESUME_FROM_CHECKPOINT ]; then
    RESUME_FROM_CHECKPOINT=""
else
    RESUME_FROM_CHECKPOINT="--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
fi


# accelerate launch src/train_bash.py \
# deepspeed --hostfile hostfile.txt src/train_bash.py \
deepspeed --num_gpus=$NUM_GPUS --master_port=9901 src/train_bash.py \
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
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps $SAVE_STEPS \
    --lora_rank 8 \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --plot_loss True \
    --val_size $VAL_SIZE \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --report_to wandb \
    $RUN_NAME_PAR \
    --preprocessing_num_workers 32 \
    --flash_attn \
    $RESUME_FROM_CHECKPOINT \
    --seed $SEED \