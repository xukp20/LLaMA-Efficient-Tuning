TIME=$(date "+%m-%d-%H-%M")
# DATASET=metamath_all_0322
DATASET=metamath_all_0508
TEMPLATE=vanilla

# wandb
# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890

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

# mistral 7b
MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/5e9c98b96d071dce59368012254c55b0ec6f8658"

# gemma 2b
# MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--google--gemma-2b/snapshots/9d067f00def958594aaa16b39a65b07d69ca655b"

# phi-2
# MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--microsoft--phi-2/snapshots/d3186761bf5c4409f7679359284066c25ab668ee/"
# MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--microsoft--phi-2/snapshots/710686f446f02286c858c11f052acb87c306ddd2"

# continue from checkpoint
# RESUME_FROM_CHECKPOINT="/cephfs/xukangping/root/models/stepbystep-metamath_all/mistral-vanilla-metamath_all-02-06-10-01/checkpoint-20000"

# continue from all model
# MODEL_NAME_OR_PATH="/cephfs/xukangping/root/models/stepbystep-metamath_all_0322/mistral-vanilla-metamath_all_0322-03-23-00-34"

# llama-3 
# MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b"


# phi-3
# MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/653ee820c4f2ee66427e997b4a8ca3e9323e7d46"

MAX_SAMPLES=10000000
VAL_SIZE=0.0001
SAVE_STEPS=5000
NUM_GPUS=8
# LR=5e-5
# LR=2e-6 # for conditions
LR=5e-7
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
EPOCHS=2
CUTOFF_LEN=4096
EVAL_STEPS=500

# nccl
# NCCL_DEBUG=INFO
# # 启用nccl
# NCCL_P2P_DISABLE=0
# # 启用 IB
# NCCL_IB_DISABLE=0
# # 不使用docker网口
# NCCL_SOCKET_IFNAME="^lo,docker0"
# # 使用ib网口
# NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3

RESUME_FROM_CHECKPOINT="/cephfs/xukangping/root/models/stepbystep-metamath_all_0508/vanilla-metamath_all_0508-05-08-19-26/checkpoint-50000"

# accelerate launch src/train_bash.py \
# deepspeed --hostfile hostfile.txt src/train_bash.py \
deepspeed --num_gpus $NUM_GPUS --master_port=9901 src/train_bash.py \
    --deepspeed "/cephfs/xukangping/code/LLaMA-Efficient-Tuning/ds_config.json" \
    --stage sft \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
    --do_train True \
    --overwrite_cache False \
    --finetuning_type full \
    --template $TEMPLATE \
    --dataset_dir data \
    --dataset $DATASET \
    --cutoff_len $CUTOFF_LEN \
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
    --warmup_steps 0 \
    --output_dir $OUTPUT_DIR \
    --fp16 True \
    --plot_loss True \
    --val_size $VAL_SIZE \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEPS \
    --report_to wandb \
    --flash_attn \
    --preprocessing_num_workers 32