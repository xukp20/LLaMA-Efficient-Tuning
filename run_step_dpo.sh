TIME=$(date "+%m-%d-%H-%M")
DATASET=step_dpo_05_3
TEMPLATE=vanilla

export WANDB_PROJECT=stepbystep-dpo-$DATASET

OUTPUT_DIR=/cephfs/xukangping/root/models/$WANDB_PROJECT/dpo-$TEMPLATE-$DATASET-$TIME

# from metamath_all_0322 model
MODEL_NAME_OR_PATH=/cephfs/xukangping/root/models/stepbystep-metamath_all_0322/mistral-vanilla-metamath_all_0322-03-23-00-34

MAX_SAMPLES=10000000
VAL_SIZE=0.0001
SAVE_STEPS=5000
NUM_GPUS=8
# LR=5e-5
# LR=2e-6 # for conditions
LR=2e-6
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
EPOCHS=3
CUTOFF_LEN=4096
EVAL_STEPS=3


deepspeed --num_gpus $NUM_GPUS --master_port=9901 src/train_bash.py \
    --deepspeed "/cephfs/xukangping/code/LLaMA-Efficient-Tuning/ds_config.json" \
    --stage dpo \
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
    --dpo_ftx 1.0 \
    --preprocessing_num_workers 32