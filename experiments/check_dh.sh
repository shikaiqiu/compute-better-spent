ALLOWED_GPUs="0 1 2 3 4 5 6 7"

project=check_dh
lr=3e-3

### Dense ####
depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
python3 train_dh.py \
--wandb_project=${project} \
--seed=${seed} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=dense \
--no-use_bias \
--scheduler=none
done;

### BTT ####
depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
python3 train_dh.py \
--wandb_project=${project} \
--seed=${seed} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=btt \
--use_wrong_mult \
--no-use_bias \
--layers=all_but_last \
--scheduler=none
done;

depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
python3 train_dh.py \
--wandb_project=${project} \
--seed=${seed} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=btt \
--no-use_bias \
--layers=all_but_last \
--scheduler=none
done;


### Monarch ####
depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
python3 train_dh.py \
--wandb_project=${project} \
--seed=${seed} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=monarch \
--num_blocks=16 \
--use_wrong_mult \
--no-use_bias \
--layers=all_but_last \
--scheduler=none
done;

depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
python3 train_dh.py \
--wandb_project=${project} \
--seed=${seed} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=monarch \
--num_blocks=16 \
--no-use_bias \
--layers=all_but_last \
--scheduler=none
done;


### Kron ####
depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
python3 train_dh.py \
--wandb_project=${project} \
--seed=${seed} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=kron \
--use_wrong_mult \
--no-use_bias \
--layers=all_but_last \
--scheduler=none
done;

depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
python3 train_dh.py \
--wandb_project=${project} \
--seed=${seed} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=kron \
--no-use_bias \
--layers=all_but_last \
--scheduler=none
done;


### Low Rank ####
depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
python3 train_dh.py \
--wandb_project=${project} \
--seed=${seed} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=low_rank \
--rank_frac=0 \
--use_wrong_mult \
--no-use_bias \
--layers=all_but_last \
--scheduler=none
done;

depth=2
width=64
for scale_factor in 0.25 1 4 16 64; do
python3 train_dh.py \
--wandb_project=${project} \
--seed=${seed} \
--dataset=cifar10 \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=1024 \
--epochs=500 \
--calculate_stats=1 \
--resolution=32 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--input_lr_mult=0.1 \
--struct=low_rank \
--rank_frac=0 \
--no-use_bias \
--layers=all_but_last \
--scheduler=none
done;