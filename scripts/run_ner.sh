set -x

export TASK_NAME=conll2003
export DATA_DIR=./data/conll2003
export OMP_NUM_THREADS=1

torchrun \
--standalone \
--nproc_per_node 4 \
  finetune_ner.py \
  --data_dir $DATA_DIR \
  --max_seq_length 2048 \
  --pretrained_checkpoint ./results/4-12-1_pretrain/pytorch_model.bin \
  --pretrain_model_config_path ./results/4-12-1_pretrain/config_model.json \
  --output_dir ./results/4-12-1_finetune_${TASK_NAME}/ \
  --char2int_dict_path ./data/char2int_dict.pkl \
  --int2char_dict_path ./data/int2char_dict.pkl \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --num_train_epochs 10 \
  --warmup_ratio 0.1 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --report_to none \
  --do_train \
  --do_eval \
  --do_predict