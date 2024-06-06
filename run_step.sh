TIME=$(date "+%m-%d-%H-%M")
DATASET=step_calculation_all
TEMPLATE=vanilla

# wandb
export WANDB_PROJECT=stepbystep-$DATASET

# set HF_HOME env
# export HF_HOME=/lustre/cache/huggingface
OUTPUT_DIR=/cephfs/xukangping/root/models/$WANDB_PROJECT/$TEMPLATE-$DATASET-$TIME
# OUTPUT_DIR=~/models/llama-tuned/codellama-34b-$DATASET-$TIME

# 7B
# origina model 7B
# MODEL_NAME_OR_PATH=/data/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
# metamath 7B
# MODEL_NAME_OR_PATH=/data/cache/huggingface/hub/models--meta-math--MetaMath-7B-V1.0/snapshots/51b13691d345ff03f2ef70f3ec1ff69ff7aeaf76
# my metamath 7B
# MODEL_NAME_OR_PATH="/root/models/llama-tuned/llama-2-7b-alpaca-meta_math-10-28-10-20"
# model added special tokens
# MODEL_NAME_OR_PATH=/data/xukp/models/llama/llama2-7b-added_special_tokens   # added special tokens

# Mistral 7B
# MODEL_NAME_OR_PATH="/data/cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/5e9c98b96d071dce59368012254c55b0ec6f8658"

# code llama
# MODEL_NAME_OR_PATH=/lustre/cache/huggingface/models--codellama--CodeLlama-34b-hf/snapshots/fda69408949a7c6689a3cf7e93e632b8e70bb8ad
# MODEL_NAME_OR_PATH="codellama/CodeLlama-34b-hf"

# llmm
# MODEL_NAME_OR_PATH="/root/models/llmm"

# llama-3 7b
# MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b"


# mistral 7b
MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/5e9c98b96d071dce59368012254c55b0ec6f8658"


VAL_SIZE=0.002
NUM_GPUS=8
# LR=5e-5
# LR=5e-7
LR=1e-6 # for conclusions
EPOCHS=2
CUTOFF_LEN=4096
SAVE_STEPS=500
EVAL_STEPS=50

# accelerate launch src/train_bash.py \
# deepspeed --hostfile hostfile.txt src/train_bash.py \
deepspeed --num_gpus $NUM_GPUS --master_port=9901 src/train_bash.py \
    --deepspeed "/cephfs/xukangping/code/LLaMAFactory/ds_config.json" \
    --stage sft \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --do_train True \
    --overwrite_cache False \
    --finetuning_type full \
    --template $TEMPLATE \
    --dataset_dir data \
    --dataset $DATASET \
    --cutoff_len $CUTOFF_LEN \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --max_samples 100000000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps $SAVE_STEPS \
    --warmup_steps 0 \
    --output_dir $OUTPUT_DIR \
    --fp16 True \
    --plot_loss True \
    --val_size $VAL_SIZE \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEPS \
    --report_to wandb \
    --flash_attn