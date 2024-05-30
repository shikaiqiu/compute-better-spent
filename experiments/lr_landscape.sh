project=lr_landscape

### Dense ####
depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
for lr in 3e-4 1e-3 3e-3 1e-2 3e-2; do
python3 train_cifar.py \
--wandb_project=${project} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=100 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=dense \
--scheduler=none
done;
done;

### BTT ####
depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
for lr in 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1; do
python3 train_cifar.py \
--wandb_project=${project} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=100 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=btt \
--use_wrong_mult \
--layers=all_but_last \
--scheduler=none
done;
done;

depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
python3 train_cifar.py \
--wandb_project=${project} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=100 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=btt \
--layers=all_but_last \
--scheduler=none
done;
done;


### Monarch ####
depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
for lr in 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1; do
python3 train_cifar.py \
--wandb_project=${project} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=100 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=monarch \
--num_blocks=16 \
--use_wrong_mult \
--layers=all_but_last \
--scheduler=none
done;
done;

depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
python3 train_cifar.py \
--wandb_project=${project} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=100 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=monarch \
--num_blocks=16 \
--layers=all_but_last \
--scheduler=none
done;
done;


### Kron ####
depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
for lr in 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1; do
python3 train_cifar.py \
--wandb_project=${project} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=100 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=kron \
--use_wrong_mult \
--layers=all_but_last \
--scheduler=none
done;
done;

depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
python3 train_cifar.py \
--wandb_project=${project} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=100 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=kron \
--layers=all_but_last \
--scheduler=none
done;
done;


### Low Rank ####
depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
for lr in 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1; do
python3 train_cifar.py \
--wandb_project=${project} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=100 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=low_rank \
--rank_frac=0 \
--use_wrong_mult \
--layers=all_but_last \
--scheduler=none
done;
done;

depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
python3 train_cifar.py \
--wandb_project=${project} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=100 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=low_rank \
--rank_frac=0 \
--layers=all_but_last \
--scheduler=none
done;
done;
