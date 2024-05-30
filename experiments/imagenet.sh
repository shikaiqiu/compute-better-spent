### do "conda activate imagenet" before running this script
export OMP_NUM_THREADS=4

USE_WANDB=1
WANDB_PROJECT=imagenet
CKPT_DIR=TODO # TODO: change to your checkpoint directory
IMAGENET_DIR=TODO # TODO: change to your imagenet directory (e.g. /data/imagenet)

LR=2e-3
EPOCHS=300
WARMUP=5
MIXUP=0.2
DROPOUT=0
DEPTH=12
WD=0.05

# Should have NUM_DEVICES * BATCH_SIZE * ACCUM = 3072
NUM_DEVICES=8
BATCH_SIZE=384
ACCUM=1

### Dense ###
for width in 80 192 384; do
python -m imagenet.train_timm \
--config-file imagenet/vit32.yaml \
--data.data_dir=${IMAGENET_DIR} \
--logging.folder=${CKPT_DIR} \
--logging.use_wandb=${USE_WANDB} \
--logging.wandb_project=${WANDB_PROJECT} \
--dist.world_size=${NUM_DEVICES} \
--training.distributed=1 \
--training.batch_size=${BATCH_SIZE} \
--training.lr=${LR} \
--training.mixup=${MIXUP} \
--training.clip_grad=1 \
--training.weight_decay=${WD} \
--training.epochs=${EPOCHS} \
--training.warmup_epochs=${WARMUP} \
--training.grad_accum_steps=${ACCUM} \
--training.schedule=cosine \
--model.patch_size=32 \
--model.width=${width} \
--model.base_depth=${DEPTH} \
--model.depth=${DEPTH} \
--model.heads=4 \
--model.dropout=${DROPOUT} \
--cola.struct=none
done

### BTT rank-1 ###
for width in 256 784 1936; do
python -m imagenet.train_timm \
--config-file imagenet/vit32.yaml \
--data.data_dir=${IMAGENET_DIR} \
--logging.folder=${CKPT_DIR} \
--logging.use_wandb=${USE_WANDB} \
--logging.wandb_project=${WANDB_PROJECT} \
--dist.world_size=${NUM_DEVICES} \
--training.distributed=1 \
--training.batch_size=${BATCH_SIZE} \
--training.lr=${LR} \
--training.mixup=${MIXUP} \
--training.clip_grad=1 \
--training.weight_decay=${WD} \
--training.epochs=${EPOCHS} \
--training.warmup_epochs=${WARMUP} \
--training.grad_accum_steps=${ACCUM} \
--training.schedule=cosine \
--model.patch_size=32 \
--model.width=${width} \
--model.base_depth=${DEPTH} \
--model.depth=${DEPTH} \
--model.heads=4 \
--model.dropout=${DROPOUT} \
--cola.struct=btt_norm
done;

### BTT rank-2 ###
for width in 144 484 1296; do
python -m imagenet.train_timm \
--config-file imagenet/vit32.yaml \
--data.data_dir=${IMAGENET_DIR} \
--logging.folder=${CKPT_DIR} \
--logging.use_wandb=${USE_WANDB} \
--logging.wandb_project=${WANDB_PROJECT} \
--dist.world_size=${NUM_DEVICES} \
--training.distributed=1 \
--training.batch_size=${BATCH_SIZE} \
--training.lr=${LR} \
--training.mixup=${MIXUP} \
--training.clip_grad=1 \
--training.weight_decay=${WD} \
--training.epochs=${EPOCHS} \
--training.warmup_epochs=${WARMUP} \
--training.grad_accum_steps=${ACCUM} \
--training.schedule=cosine \
--model.patch_size=32 \
--model.width=${width} \
--model.base_depth=${DEPTH} \
--model.depth=${DEPTH} \
--model.heads=4 \
--model.dropout=${DROPOUT} \
--cola.struct=btt_norm \
--cola.tt_rank=2
done;

### Monarch 4 blocks ###
for width in 140 316 632; do
python -m imagenet.train_timm \
--config-file imagenet/vit32.yaml \
--data.data_dir=${IMAGENET_DIR} \
--logging.folder=${CKPT_DIR} \
--logging.use_wandb=${USE_WANDB} \
--logging.wandb_project=${WANDB_PROJECT} \
--dist.world_size=${NUM_DEVICES} \
--training.distributed=1 \
--training.batch_size=${BATCH_SIZE} \
--training.lr=${LR} \
--training.mixup=${MIXUP} \
--training.clip_grad=1 \
--training.weight_decay=${WD} \
--training.epochs=${EPOCHS} \
--training.warmup_epochs=${WARMUP} \
--training.grad_accum_steps=${ACCUM} \
--training.schedule=cosine \
--model.patch_size=32 \
--model.width=${width} \
--model.base_depth=${DEPTH} \
--model.depth=${DEPTH} \
--model.heads=4 \
--model.dropout=${DROPOUT} \
--cola.struct=monarch
done;

### Monarch 16 blocks ###
for width in 280 632 1264; do
python -m imagenet.train_timm \
--config-file imagenet/vit32.yaml \
--data.data_dir=${IMAGENET_DIR} \
--logging.folder=${CKPT_DIR} \
--logging.use_wandb=${USE_WANDB} \
--logging.wandb_project=${WANDB_PROJECT} \
--dist.world_size=${NUM_DEVICES} \
--training.distributed=1 \
--training.batch_size=${BATCH_SIZE} \
--training.lr=${LR} \
--training.mixup=${MIXUP} \
--training.clip_grad=1 \
--training.weight_decay=${WD} \
--training.epochs=${EPOCHS} \
--training.warmup_epochs=${WARMUP} \
--training.grad_accum_steps=${ACCUM} \
--training.schedule=cosine \
--model.patch_size=32 \
--model.width=${width} \
--model.base_depth=${DEPTH} \
--model.depth=${DEPTH} \
--model.heads=4 \
--model.dropout=${DROPOUT} \
--cola.struct=monarch \
--cola.num_blocks=16
done;
