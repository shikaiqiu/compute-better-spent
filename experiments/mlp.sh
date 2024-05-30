ds=cifar10 # choose from {cifar10, cifar100}
lr=3e-3

### Dense ####
depth=3
width=64
struct=dense
for scale_factor in 0.5 1 2 4 8 16 32 64; do
python3 train_cifar.py \
--wandb_project=mlp_${ds} \
--dataset=${ds} \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=${struct} \
--scheduler=cosine
done;

### Kron ####
depth=3
width=64
struct=kron
layers=all_but_last
for scale_factor in 2 4 8 16 32 32 64 128 256; do
python3 train_cifar.py \
--wandb_project=mlp_${ds} \
--dataset=${ds} \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=${struct} \
--layers=${layers} \
--scheduler=cosine
done;

### Low Rank ####
depth=3
width=64
struct=low_rank
layers=all_but_last
for scale_factor in 2 4 8 16 32 64 128 256; do
python3 train_cifar.py \
--wandb_project=mlp_${ds} \
--dataset=${ds} \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=${struct} \
--layers=${layers} \
--scheduler=cosine
done;


### BTT ####
depth=3
width=64
struct=btt
layers=all_but_last
for scale_factor in 2 4 8 16 32 64 128 256; do
python3 train_cifar.py \
--wandb_project=mlp_${ds} \
--dataset=${ds} \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=${struct} \
--layers=${layers} \
--scheduler=cosine
done;

### Monarch ####
depth=3
width=64
struct=monarch
layers=all_but_last
for scale_factor in 0.7 1.4 2.8 5.6 11 22 45 90; do
python3 train_cifar.py \
--wandb_project=mlp_${ds} \
--dataset=${ds} \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=${struct} \
--layers=${layers} \
--scheduler=cosine
done;

### TT ####
depth=3
width=64
struct=tt
layers=all_but_last
for scale_factor in 0.25 0.5 1 2 4 8 16 32 64; do
python3 train_cifar.py \
--wandb_project=mlp_${ds} \
--dataset=${ds} \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=${struct} \
--layers=${layers} \
--tt_rank=16 \
--scheduler=cosine
done;
