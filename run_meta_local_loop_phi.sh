TIME=$(date "+%m-%d-%H-%M")
DATASET=metamathQA
TEMPLATE=alpaca

# wandb
export WANDB_PROJECT=xukp20-$DATASET-local-loop

PROJECTION_TYPE=$1
LOOP_LAYERS=$2
LOOP_TIMES=$3
FIX_PROJECTION=$4
PROJECTION_INIT=$5
FIX_LLM=$6

RUN_NAME=${PROJECTION_TYPE}_${LOOP_LAYERS}_loop${LOOP_TIMES}_fix-p${FIX_PROJECTION}_${PROJECTION_INIT}_fix-llm${FIX_LLM}
if [ -z $RUN_NAME ]; then
    RUN_NAME_PAR=""
else
    RUN_NAME_PAR="--run_name $RUN_NAME"
fi

SEED=$7
if [ -z $SEED ]; then
    SEED=42
fi

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
# model with trainable vec alpha
MODEL_NAME_OR_PATH="/cephfs/xukangping/root/models/xukp20-metamathQA-local-loop/alpaca-metamathQA-06-03-10-30"
BASE_MODEL=0

# pass params through env
export CUSTOM_MODEL_PATH="/cephfs/xukangping/code/LLaMA-Efficient-Tuning/load_my_model.py"

export MODEL_PATH=$MODEL_NAME_OR_PATH
export PROJECTION_TYPE=$PROJECTION_TYPE
export LOOP_LAYERS=$LOOP_LAYERS
export LOOP_TIMES=$LOOP_TIMES
export FIX_PROJECTION=$FIX_PROJECTION
export PROJECTION_INIT=$PROJECTION_INIT
export FIX_LLM=$FIX_LLM
export BASE_MODEL=$BASE_MODEL

VAL_SIZE=0.005
NUM_GPUS=8
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
# LR=5e-5
LR=1e-5 # for meta_math
EPOCHS=3
MAX_LEN=2048
SAVE_STEPS=3000

if [ $FIX_LLM == 1 ]; then
    LR=1e-4
    EPOCHS=2
fi

# for continue the gate training
# VAL_SIZE=0.005
# NUM_GPUS=8
# BATCH_SIZE=4
# # LR=5e-5
# LR=2e-6 # for meta_math
# EPOCHS=4
# MAX_LEN=2048
# SAVE_STEPS=3000

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
    # --load_best_model_at_end True