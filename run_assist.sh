TIME=$(date "+%m-%d-%H-%M")
TASK=hint_summary
DATASET=assist_$TASK
TEMPLATE=vanilla

# wandb
export WANDB_PROJECT=xukp20-assist-$TASK


BASE_MODEL=mistral
# BASE_MODEL=llama

# set HF_HOME env
# export HF_HOME=/lustre/cache/huggingface
if [ $BASE_MODEL == "mistral" ]; then
    OUTPUT_DIR=/cephfs/xukangping/root/models/assist/mistral-7b-$TEMPLATE-$DATASET-$TIME
    MODEL_NAME_OR_PATH="/cephfs/shared/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/5e9c98b96d071dce59368012254c55b0ec6f8658"
elif [ $BASE_MODEL == "llama" ]; then
    OUTPUT_DIR=/cephfs/xukangping/root/models/assist/llama-7b-$TEMPLATE-$DATASET-$TIME
    MODEL_NAME_OR_PATH=/data/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
fi

VAL_SIZE=0.002
NUM_GPUS=8
# LR=5e-5
LR=5e-7
# LR=2e-6
EPOCHS=3
CUTOFF_LEN=4096

# accelerate launch src/train_bash.py \
# deepspeed --hostfile hostfile.txt src/train_bash.py \
deepspeed --num_gpus $NUM_GPUS --master_port=9901 src/train_bash.py \
    --deepspeed "/cephfs/xukangping/code/LLaMA-Efficient-Tuning/ds_config.json" \
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
    --max_samples 1000000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --plot_loss True \
    --val_size $VAL_SIZE \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --report_to wandb \
    --flash_attn 
    # --load_best_model_at_end True
