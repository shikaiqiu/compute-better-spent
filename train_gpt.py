"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
from contextlib import nullcontext
from datetime import datetime
import numpy as np
import torch
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from nn.gpt2 import GPT
from model.gpt_fns import get_lr_mult
from model.gpt_fns import update_lrs
from model.gpt_fns import reset_lrs
from model.gpt_fns import get_batch_all
from model.gpt_fns import estimate_loss
from model.gpt_fns import get_epoch
from model.gpt_fns import init_dist_process
from model.gpt_fns import construct_configs
from nn.cola_nn import cola_parameterize
from nn.cola_nn import get_model_summary_and_flops
from tqdm import tqdm

# -----------------------------------------------------------------------------
eval_interval, log_interval, eval_iters, eval_only = 2000, 1, 200, False
out_dir, always_save_checkpoint = "out", True
ckpt_path = ""
dataset = 'open'
data_dir = 'data/'
data_dir = os.path.join(data_dir, dataset)
block_size, batch_size, gradient_accumulation_steps = 1024, 12, 5 * 8
vocab_size = 50_304
data_dtype = np.uint16
base_n_head, base_d_head, base_d_model, base_d_embd, base_ffn_expansion = -1, 64, 768, 768, 1
n_head, d_head, d_model, d_embd, ffn_expansion = -1, 64, 768, -1, 4
n_layer = 12
split_qkv = False
axial = False
dropout, bias, do_qk_ln = 0.0, False, False
opt_name, init_lr, weight_decay, beta1, beta2, grad_clip = "AdamW", 6e-4, 1e-1, 0.9, 0.95, 1.0
decay_lr, warmup_iters = True, 2_000
spec_penalty_weight = 0.
aux_loss_weight = 0.01
max_iters = 600_000
backend = 'nccl'  # 'nccl', 'gloo', etc.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
struct = "none"
tt_cores, tt_rank, num_blocks, rank_frac, every_n_fwds = 2, 1, 4, 0.2, 100
lm_head_struct = ''
lm_head_tt_rank = -1
lm_head_rank_frac = -1.
layers, input_lr_mult = "all_but_last", 1.
wandb_log, wandb_project = False, "attention"
# -----------------------------------------------------------------------------
exec(open('./model/configurator.py').read())  # overrides from command line or config file
if d_model != d_embd and d_embd != -1:
    base_d_embd = base_d_model - 1  # a hack to ensure base model has emb up/down sampler params
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H%M%S")
lr_decay_iters, min_lr = max_iters, init_lr / 10.
# wandb_run_name = f"l{n_layer}-h{n_head}-d{d_model}-e{d_embd}-{struct}_{now.strftime('%H%M%S')}"
wandb_run_name = f"{struct}_{layers}_l{n_layer}-dm{d_model}-de{d_embd}-h{n_head}-dh{d_head}-ttr{tt_rank}-{now.strftime('%H%M%S')}"
out_dir = f'{out_dir}/{wandb_run_name}'
device_type = 'cuda' if 'cuda' in device else 'cpu'
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
print("*=" * 50)
for key in sorted(config.keys()):
    print(f"{key}: {config[key]}")
print("*=" * 50)
# -----------------------------------------------------------------------------
print(f"Eval interval: {eval_interval:,d}")

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    aux = init_dist_process(backend, gradient_accumulation_steps)
    master_process, seed_offset, ddp_world_size, ddp_local_rank, gradient_accumulation_steps = aux
else:
    master_process, seed_offset, ddp_world_size, ddp_local_rank = True, 0, 1, 0
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=data_dtype, mode='r')
dataset_size = len(train_data)
print(f"Total tokens: {dataset_size:,d}")
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=data_dtype, mode='r')

if len(ckpt_path) == 0:
    base_config, target_config, cola_kwargs, optim_kwargs = construct_configs(**config)
    model, optimizer = cola_parameterize(GPT, base_config, init_lr, target_config=target_config, struct=struct,
                                         layer_select_fn=layers, device=device, cola_kwargs=cola_kwargs,
                                         optim_kwargs=optim_kwargs)
    iter_num, best_val_loss = 1, 1e9
else:
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    config["out_dir"] = out_dir
    config["data_dir"] = data_dir
    base_config, target_config, cola_kwargs, optim_kwargs = construct_configs(**ckpt['config'])
    init_lr, struct, layers, device = config["init_lr"], config["struct"], config["layers"], config["device"]
    model, optimizer = cola_parameterize(GPT, base_config, init_lr, target_config=target_config, struct=struct,
                                         layer_select_fn=layers, device=device, cola_kwargs=cola_kwargs,
                                         optim_kwargs=optim_kwargs)
    model.load_state_dict(ckpt["model"])
    iter_num, best_val_loss = ckpt["iter_num"], ckpt["best_val_loss"]
    for key, value in config.items():
        if key in globals():
            globals()[key] = value

fake_input = torch.randint(low=0, high=vocab_size, size=(1, block_size)).to(device)
info = get_model_summary_and_flops(model, (fake_input, fake_input))
emb_params = sum([p.numel() for name, p in model.named_parameters() if 'wte' in name or 'wpe' in name])
head_params = sum([p.numel() for name, p in model.named_parameters() if 'lm_head' in name])
info['emb_params'] = emb_params
info['head_params'] = head_params
info['non_emb_params'] = info['cola_params'] - emb_params - head_params # i.e. non-embedding params
param_str = f'P: {info["cola_params"]/1e6:.2f} M | E: {emb_params/1e6:.2f} M | H: {head_params/1e6:.2f} M |'
param_str += f' Non-embd: {info["non_emb_params"]/1e6:.2f} M'
flops = info['cola_flops']
flops_per_token = flops / block_size
non_emb_flops = flops - head_params * block_size  # exclude emb and unemb
non_emb_flops_per_token = non_emb_flops / block_size
info['non_emb_flops'] = non_emb_flops
print(param_str)
print(f'Non-emb FLOPs: {non_emb_flops // 1e6} M')
config.update(info)

# write config to log.txt one variable per line
if master_process:
    with open(os.path.join(out_dir, 'log.txt'), 'w') as f:
        for key in sorted(config.keys()):
            f.write(f"{key}: {config[key]}\n")
# optimizer = model.configure_optimizers(weight_decay, init_lr, (beta1, beta2), device_type)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

get_batch = partial(get_batch_all, train_data=train_data, val_data=val_data, batch_size=batch_size, block_size=block_size,
                    device=device, device_type=device_type)

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

estimate_loss = partial(estimate_loss, model=model, get_batch=get_batch, eval_iters=eval_iters, ctx=ctx, block_size=block_size)
get_epoch = partial(get_epoch, dataset_size=dataset_size, block_size=block_size, batch_size=batch_size,
                    num_devices=ddp_world_size)

if wandb_log and master_process:
    import wandb
    wandb.init(entity="ap-team", project=wandb_project, name=wandb_run_name, config=config)

ckpt_count, ckpt_max = 1, 100
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
init_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

start_time = time.time()

for _ in (pbar := tqdm(range(max_iters))):
    mult = get_lr_mult(iter_num, init_lr, min_lr if decay_lr else init_lr, warmup_iters, lr_decay_iters)
    global_lr = init_lr * mult
    update_lrs(optimizer, mult)

    if (iter_num % int(eval_interval) == 0 or eval_only) and master_process:
        epoch_num = get_epoch(iter_num)
        losses = estimate_loss()
        if wandb_log:
            metrics = {
                "epoch": epoch_num,
                "iter": iter_num,
                "compute": flops_per_token * tokens_per_iter * iter_num,
                "non_emb_compute": non_emb_flops_per_token * tokens_per_iter * iter_num,
                "train/loss": losses['train'][0],
                "train/ppl": losses['train'][-1],
                "val/loss": losses['val'][0],
                "val/ppl": losses['val'][-1],
                "lr": global_lr,
                "mfu": running_mfu * 100,
            }
            for name, p in model.named_parameters():
                if 'top_singular_vec' in name:
                    metrics[f'v_rms/{name}'] = torch.sqrt((p**2).mean()).item()
                if hasattr(p, 'out'):
                    metrics[f'out/{name}'] = p.out
                if hasattr(p, 'scale'):
                    metrics[f'scale/{name}'] = p.scale
                if hasattr(p, 'rms'):
                    metrics[f'rms/{name}'] = p.rms
                if hasattr(p, 'ppl'):
                    metrics[f'ppl/{name}'] = p.ppl
                if hasattr(p, 'agg_ppl'):
                    metrics[f'agg_ppl/{name}'] = p.agg_ppl
            for name, p in model.named_modules():
                if hasattr(p, 'natural_norm'):
                    metrics[f'natural_norm/{name}'] = p.natural_norm
            if hasattr(raw_model, 'get_features'):
                hs = raw_model.get_features()
                raw_model.clear_features()
                for i, h in enumerate(hs):
                    metrics[f'hs/{i}'] = h
            wandb.log(metrics)
        pb = f"I {iter_num} | L {losses['val'][0]:.2f}"
        # open log.txt and write val loss
        with open(os.path.join(out_dir, 'log.txt'), 'a') as f:
            log_str = f'I {iter_num} | L {losses["val"][0]:.4f} | P {losses["val"][-1]:1.3e}'
            log_str += f' | Lt {losses["train"][0]:.4f} | Pt {losses["train"][-1]:1.3e}'
            log_str += f' | H {hs[-2]:.4f}'
            eta = (time.time() - start_time) / (iter_num + 1) * (max_iters - iter_num)
            log_str += f' | ETA {eta/3600:.2f}h\n'
            f.write(log_str)
        if ckpt_count <= ckpt_max:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'base_config': base_config,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H%M%S") + now.strftime("-%f")[:4]
            torch.save(checkpoint, os.path.join(out_dir, f"ckpt_{timestamp}.pt"))
            ckpt_count += 1
        if (losses['val'][0] < best_val_loss or always_save_checkpoint) and (iter_num > 0):
            best_val_loss = losses['val'][0]
            print(f"saving best checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))(model, out_dir)(model, out_dir)
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss, aux_loss, spectral_penalty = model(X, Y)
            loss = (loss + aux_loss_weight * aux_loss + spec_penalty_weight * spectral_penalty) / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    reset_lrs(optimizer, init_lrs)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.2f}, spec {spectral_penalty:.2f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1
    pbar.update(1)

    if iter_num > max_iters:
        # write in master process
        if master_process:
            with open(os.path.join(out_dir, 'log.txt'), 'a') as f:
                # log the losses
                log_str = f'I {iter_num} | L {losses["val"][0]:.4f} | P {losses["val"][-1]:1.3e}'
                log_str += f' | Lt {losses["train"][0]:.4f} | Pt {losses["train"][-1]:1.3e}\n'
                f.write(log_str)
                f.write('Finished training!\n')
        break

if ddp:
    destroy_process_group()
