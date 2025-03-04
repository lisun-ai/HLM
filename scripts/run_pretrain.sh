set -x

export OMP_NUM_THREADS=1

torchrun \
   --standalone \
   --nproc_per_node 8 \
   pretrain.py \
    --config_json ./config/config_pretrain.json