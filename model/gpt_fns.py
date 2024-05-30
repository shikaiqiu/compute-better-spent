import os
import math
import torch
from torch.distributed import init_process_group
import numpy as np


def construct_configs(
    n_layer,
    base_n_head,
    base_d_head,
    base_d_model,
    base_d_embd,
    n_head,
    d_head,
    d_model,
    d_embd,
    block_size,
    bias,
    vocab_size,
    dropout,
    do_qk_ln,
    split_qkv,
    base_ffn_expansion,
    ffn_expansion,
    struct,
    tt_cores,
    tt_rank,
    num_blocks,
    rank_frac,
    every_n_fwds,
    opt_name,
    weight_decay,
    init_lr,
    beta1,
    beta2,
    device,
    axial=False,
    lm_head_struct=None,
    lm_head_tt_rank=None,
    lm_head_rank_frac=None,
    **_,
):
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    base_config = dict(n_layer=n_layer, n_head=base_n_head, d_head=base_d_head, d_model=base_d_model, d_embd=base_d_embd,
                       block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout, do_qk_ln=do_qk_ln,
                       split_qkv=split_qkv, ffn_expansion=base_ffn_expansion, axial=False)
    target_config = base_config.copy()
    target_config['n_head'] = n_head
    target_config['d_head'] = d_head
    target_config['d_model'] = d_model
    target_config['d_embd'] = d_embd
    target_config['ffn_expansion'] = ffn_expansion
    target_config['axial'] = axial
    if lm_head_struct is None:
        lm_head_struct = struct
        lm_head_tt_rank = tt_rank
        lm_head_rank_frac = rank_frac
    cola_kwargs = dict(tt_cores=tt_cores, tt_rank=tt_rank, num_blocks=num_blocks, rank_frac=rank_frac, every_n_fwds=every_n_fwds,
                       do_qk_ln=do_qk_ln, lm_head_struct=lm_head_struct, lm_head_tt_rank=lm_head_tt_rank,
                       lm_head_rank_frac=lm_head_rank_frac)
    optim_kwargs = {
        "opt_name": opt_name,
        "weight_decay": weight_decay,
        "lr": init_lr,
        "betas": (beta1, beta2),
        "device_type": device_type
    }
    return base_config, target_config, cola_kwargs, optim_kwargs


def init_dist_process(backend, gradient_accumulation_steps):
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    print(f"World Size: {ddp_world_size:,d}")
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
    return master_process, seed_offset, ddp_world_size, ddp_local_rank, gradient_accumulation_steps


def get_epoch(iter_num, dataset_size, block_size, batch_size, num_devices):
    denom = dataset_size / (block_size * batch_size * num_devices)
    return int(iter_num // denom)


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters, ctx, block_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        nll_total, batches_n = 0, 0
        for k in range(eval_iters):
            X, Y = get_batch(split)
            batches_n += X.shape[0]
            with ctx:
                logits, loss, _, _ = model(X, Y)
            losses[k] = loss.item()
            # log_probs = torch.log_softmax(logits.double(), dim=-1)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            # log_probs = torch.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(-1, Y.unsqueeze(-1))
            nll_total += -target_log_probs.sum()
            batches_n += X.shape[0] * block_size
        out[split] = (losses.mean(), torch.exp(nll_total / batches_n))
    model.train()
    return out


def get_batch_all(split, train_data, val_data, batch_size, block_size, device, device_type):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def update_lrs(optimizer, mult):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= mult


def reset_lrs(optimizer, lrs):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lrs[idx]


def get_lr_mult(it, init_lr, min_lr, warmup_iters, lr_decay_iters):
    if it < warmup_iters:
        return it / warmup_iters
    if it > lr_decay_iters:
        return min_lr / init_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    ratio = min_lr / init_lr
    return ratio + coeff * (1 - ratio)
