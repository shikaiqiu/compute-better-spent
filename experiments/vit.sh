ds=cifar100
lr=1e-3

### Dense ####
depth=3
width=64
struct=dense
for scale_factor in 0.5 1 2 4 6 8 10 12; do
python3 train_cifar.py \
--wandb_project=vit_${ds} \
--dataset=${ds} \
--model=ViT \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=256 \
--epochs=200 \
--resolution=32 \
--patch_size=8 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--struct=${struct} \
--scheduler=cosine
done;

### Kron ####
depth=3
width=64
struct=kron
layers=all_but_last
for scale_factor in 1 2 4 8 16 32 64; do
python3 train_cifar.py \
--wandb_project=vit_${ds} \
--dataset=${ds} \
--model=ViT \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=256 \
--epochs=200 \
--resolution=32 \
--patch_size=8 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--struct=${struct} \
--layers=${layers} \
--scheduler=cosine
done;

### Low Rank ####
depth=3
width=64
struct=low_rank
layers=intermediate
for scale_factor in 1 2 4 8 16 32; do
python3 train_cifar.py \
--wandb_project=vit_${ds} \
--dataset=${ds} \
--model=ViT \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=256 \
--epochs=200 \
--resolution=32 \
--patch_size=8 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--struct=${struct} \
--rank_frac=0.1 \
--layers=${layers} \
--scheduler=cosine
done;

### BTT ####
depth=3
width=64
struct=btt
layers=all_but_last
for scale_factor in 1 2 4 8 16 32 64; do
python3 train_cifar.py \
--wandb_project=vit_${ds} \
--dataset=${ds} \
--model=ViT \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=256 \
--epochs=200 \
--resolution=32 \
--patch_size=8 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--struct=${struct} \
--layers=${layers} \
--scheduler=cosine
done;


# ### Monarch ####
depth=3
width=64
struct=monarch
layers=all_but_last
for scale_factor in 0.5 1.5 4 8 12 16; do
python3 train_cifar.py \
--wandb_project=vit_${ds} \
--dataset=${ds} \
--model=ViT \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=256 \
--epochs=200 \
--resolution=32 \
--patch_size=8 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--struct=${struct} \
--layers=${layers} \
--scheduler=cosine
done;

### TT ####
depth=3
width=64
struct=tt
layers=all_but_last
for scale_factor in 0.5 1 2 4 8 16 32; do
python3 train_cifar.py \
--wandb_project=vit_${ds} \
--dataset=${ds} \
--model=ViT \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=256 \
--epochs=200 \
--resolution=32 \
--patch_size=8 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--struct=${struct} \
--tt_rank=8 \
--layers=${layers} \
--scheduler=cosine
done;
