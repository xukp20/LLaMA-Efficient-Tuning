TIME=$(date "+%m-%d-%H-%M")
DATASET_NAME=$1
TYPE=$2

MODEL_NAME=$3

if [ $MODEL_NAME == "base" ]; then
    MODEL_NAME_OR_PATH=/cephfs/xukangping/root/models/small_transformers/mistral-tiny
fi

MAX_SAMPLES=$4
if [ -z $MAX_SAMPLES ]; then
    MAX_SAMPLES=100000000
fi

# arithemetics_dataset_name_type
DATASET=arithmetics_${DATASET_NAME}_${TYPE}
TEMPLATE=vanilla

# wandb
export WANDB_PROJECT=arithmetics

RUN_NAME=${DATASET_NAME}_${TYPE}_${MODEL_NAME}_${MAX_SAMPLES}
if [ -z $RUN_NAME ]; then
    RUN_NAME_PAR=""
else
    RUN_NAME_PAR="--run_name $RUN_NAME"
fi

SEED=$5
if [ -z $SEED ]; then
    SEED=42
fi

# set HF_HOME env
# export HF_HOME=/lustre/cache/huggingface
OUTPUT_DIR=/cephfs/xukangping/root/models/$WANDB_PROJECT/$TEMPLATE-$DATASET-$TIME
# OUTPUT_DIR=~/models/llama-tuned/codellama-34b-$DATASET-$TIME

VAL_SIZE=0.005
NUM_GPUS=1
BATCH_SIZE=256
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-4 # pretraining
EPOCHS=1
MAX_LEN=512
SAVE_STEPS=3000
EVAL_STEPS=30

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
    --max_samples $MAX_SAMPLES \
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
    --eval_steps $EVAL_STEPS \
    --report_to wandb \
    $RUN_NAME_PAR \
    --preprocessing_num_workers 32 \
    --flash_attn \
    $RESUME_FROM_CHECKPOINT \
    --seed $SEED \
    # --load_best_model_at_end True