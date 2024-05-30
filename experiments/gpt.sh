DATA_DIR=open # TODO
OUT_DIR=./ # TODO

# Keep product at 480
BATCH_SIZE=12
GRAD_ACCUM=40
MAX_ITERS=600_000

LR=2e-3

tt_rank=4
for d_model in 1024 1536 2048 2560; do
torchrun --nproc_per_node=8 --master_port=$(shuf -i 49152-65535 -n 1) train_gpt.py config/train_open.py --struct=btt_norm --layers=all --d_model=${d_model} --tt_rank=${tt_rank} --n_layer=12 --n_head=6 --d_head=64 --max_iters=${MAX_ITERS} --data_dir=${DATA_DIR} --out_dir=${OUT_DIR} --batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --init_lr=${LR}
done;

for d_model in 384 512 768; do
torchrun --nproc_per_node=8 --master_port=$(shuf -i 49152-65535 -n 1) train_gpt.py config/train_open.py --struct=dense --layers=all --d_model=${d_model} --n_layer=12 --n_head=-1 --d_head=64 --max_iters=${MAX_ITERS} --data_dir=${DATA_DIR} --out_dir=${OUT_DIR} --batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --init_lr=${LR}
done;