set -x

export OMP_NUM_THREADS=1
export TASK_NAME=mrpc

torchrun \
--standalone \
--nproc_per_node 4 \
   finetune_glue.py \
    --task_name $TASK_NAME \
    --output_dir ./results/4-12-1_finetune_${TASK_NAME} \
    --pretrained_checkpoint ./results/4-12-1_pretrain/pytorch_model.bin \
    --model_name_or_path char2word \
    --relative_attention True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2.5e-5 \
    --hidden_size 768 \
    --attention_heads 12 \
    --transformer_ff_size 3072 \
    --local_transformer_ff_size 1536 \
    --dropout 0.1 \
    --activation gelu \
    --max_char_length 2048 \
    --max_char_per_word 20 \
    --max_num_word 1024 \
    --n_local_layer_first 4 \
    --n_global_layer 12 \
    --n_local_layer_last 0 \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 \
    --group_by_length False \
    --logging_steps 500 \
    --save_total_limit 1 \
    --char2int_dict ./data/char2int_dict.pkl \
    --int2char_dict ./data/int2char_dict.pkl \
    --use_token_type False \
    --use_projection True \
    --word_context 1 \
    --report_to "none"
    #--noise_method random_case \