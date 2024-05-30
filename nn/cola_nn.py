from functools import reduce
import inspect
from math import prod
from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary
from sympy import factorint
from cola.ops import Dense
from ops.operators import TT, FasterBTT, Monarch
from ops.operators import BTT, Permutation

try:
    from torchdistx.deferred_init import deferred_init, materialize_module
except ImportError:

    def deferred_init(builder, **kwargs):
        return builder(**kwargs)

    def materialize_module(*args, **kwargs):
        return None

    print("Unable to lazy init models")


def is_cola_param(x):
    if isinstance(x, torch.Tensor) or isinstance(x, nn.Parameter):
        return x.dtype == torch.float32 or x.dtype == torch.float64 or x.dtype == torch.float16
    return False


def dense_init(linear, zero_init=False):
    d_in, _ = linear.in_features, linear.out_features
    if zero_init:
        std = 0
        linear.weight.zero_init = True
    else:
        std = d_in**-0.5
    linear.weight.data.normal_(mean=0, std=std)
    if linear.bias is not None:
        linear.bias.data.zero_()

@torch.no_grad()
def cola_init(tensor, zero_init=False, init_method='µP'):
    assert hasattr(tensor, 'd_in') and hasattr(tensor, 'd_out'), f'Learnable CoLA parameter {tensor} must have d_in and d_out attributes'
    if zero_init:
        print(f'Zero init cola param of shape {list(tensor.shape)}')
        return tensor.zero_()
    else:
        if init_method == 'standard':
            std = np.sqrt(1 / tensor.d_in)
            tensor.normal_(0, std)
        elif init_method == 'µP':
            # sqrt(d_out / d_in) spectral norm
            std = np.sqrt(1 / tensor.d_in) * min(1, np.sqrt(tensor.d_out / tensor.d_in))
            tensor.normal_(0, std)
        else:
            raise ValueError(f'Unknown init method {init_method}')

def factorize(x, n):
    # Find the most even way to factorize x into n integers
    # Algorithm is greedy, so it may not be optimal
    
    # Get prime factors and their counts
    prime_factors = factorint(x)
    # Initialize the n integers
    numbers = [1] * n
    # Distribute the prime factors
    for prime, count in prime_factors.items():
        for _ in range(count):
            # Find the number with the smallest product to assign the prime factor
            min_index = min(range(n), key=lambda i: numbers[i])
            numbers[min_index] *= prime
    # return in ascending order
    return sorted(numbers)

class CoLALayer(nn.Module):
    def __init__(self, A, bias=True):
        super().__init__()
        d_in_orig, d_out_orig = A.shape
        cola_tensors, self.unflatten = A.flatten()
        # learnable matrices parametrizing A
        self.op_params = nn.ParameterList()
        num_matrices = sum(is_cola_param(t) and t.dim() > 0 for t in cola_tensors)  # ignore scalars
        # Iterate over cola_tensors and only turn those that meet the condition into Parameters
        # e.g. permutations are excluded
        self.cola_tensors = []
        for t in cola_tensors:
            if is_cola_param(t):
                if t.dim() > 0:
                    assert hasattr(t, 'd_in'), f'Learnable CoLA parameter {t} must have d_in attribute'
                    d_in = t.d_in
                    d_out = t.d_out
                    if hasattr(t, "lr_mult"):
                        lr_mult = t.lr_mult
                    else:
                        # a heuristic to keep feature updates of similar size compared to a dense layer
                        lr_mult = d_in_orig / d_in / num_matrices
                    t = nn.Parameter(t)
                    t.d_in = d_in
                    t.d_out = d_out
                    t.lr_mult = lr_mult
                else:
                    t = nn.Parameter(t)
                    t.lr_mult = None
                self.cola_tensors.append(t)
                self.op_params.append(t)
            else:
                self.cola_tensors.append(t)

        self.in_features = d_in_orig
        self.out_features = d_out_orig
        self.A = self.unflatten(self.cola_tensors)  # (d_in, d_out)
        self.b = nn.Parameter(torch.zeros(d_out_orig)) if bias else None
        if bias:
            self.b.lr_mult = 1
        self.info = {}

    def _apply(self, fn):
        # _apply is called when doing model.to(device), model.cuda(), model.float(), etc.
        # apply fn to all parameters
        super()._apply(fn)  # TODO: check what happens to non-parameter tensors? (e.g. do they get moved to device?)
        # reconstruct A
        self.A = self.unflatten(self.cola_tensors)
        return self

    def update_info(self, x, out):
        with torch.no_grad():
            p = self.op_params[0]
            p.x = torch.sqrt(torch.mean(x**2) + 1e-8).item()
            p.out = torch.sqrt(torch.mean(out**2) + 1e-8).item()
            p.scale = p.out / p.x
            for p in self.op_params:
                if p.dim() > 0:
                    p.rms = normalized_rms(p)
                else:
                    p.rms = p.abs().item()
    
    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1]) # (B, d)
        if (x.shape[-1] != self.A.shape[0]) and self.A.allow_padding:
            x_in, A_in = x.shape[-1], self.A.shape[0]
            if A_in >= x_in:
                x = F.pad(x, pad=(0, self.A.shape[0] - x.shape[-1]))
            else:
                x = x[:, :A_in]
        out = x @ self.A
        if self.b is not None:
            out = out + self.b
        if not self.training:
            self.update_info(x, out)
        return out.view(*batch_shape, -1)

def build_dense(d_in, d_out, bias=True, zero_init=False, init_method='µP', **_):
    U = torch.randn(d_in, d_out)
    U.d_in, U.d_out = d_in, d_out
    U.in_dims, U.out_dims = (0,), (1,)
    cola_init(U, zero_init, init_method=init_method)
    return CoLALayer(Dense(U), bias=bias)

def build_low_rank(d_in, d_out, rank_frac=0, bias=True, zero_init=False, init_method='µP', **_):
    assert rank_frac >= 0, 'rank_frac must be non-negative'
    if rank_frac == 0:
        rank_frac = 1 / np.sqrt(min(d_in, d_out))
    rank = ceil(rank_frac * min(d_in, d_out))
    U = torch.randn(d_in, rank)
    U.d_in, U.d_out = d_in, rank
    U.in_dims, U.out_dims = (0,), (1,)
    V = torch.randn(rank, d_out)
    V.d_in, V.d_out = rank, d_out
    V.in_dims, V.out_dims = (0,), (1,)
    cola_init(U, init_method=init_method)
    cola_init(V, zero_init, init_method=init_method)
    A = Dense(U) @ Dense(V)
    return CoLALayer(A, bias=bias)

def build_tt(d_in, d_out, tt_cores, tt_rank, permute=False, bias=True, zero_init=False, init_method='µP', **_):
    ns, ms = factorize(d_in, tt_cores), factorize(d_out, tt_cores)
    print(f'TT shape: {ns} -> {ms}')
    cores = []
    for idx in range(tt_cores):
        rank_prev = 1 if idx == 0 else tt_rank
        rank_next = 1 if idx == tt_cores - 1 else tt_rank
        core = torch.randn(rank_prev, ns[idx], ms[idx], rank_next)
        core.d_in = ns[idx] * rank_prev
        core.d_out = ms[idx] * rank_next
        core.in_dims = (0, 1)
        core.out_dims = (2, 3)
        cola_init(core, zero_init and idx == tt_cores - 1, init_method=init_method)
        cores.append(core)

    A = TT(cores)
    if permute:
        P_in = Permutation(torch.randperm(d_in), dtype=A.dtype)
        P_out = Permutation(torch.randperm(d_out), dtype=A.dtype)
        A = P_in @ A @ P_out
    return CoLALayer(A, bias=bias)

def build_btt(d_in, d_out, tt_cores=2, tt_rank=1, bias=True, permute=False, zero_init=False, normalize=False, learn_gamma=True,
                  init_method='µP', **_):
    assert tt_rank > 0, 'tt_rank must be positive'
    ns, ms = tuple(factorize(d_out, tt_cores)), tuple(factorize(d_in, tt_cores))
    rs = (1, tt_rank, 1)
    shapes = (rs, ms, ns)
    cores = []
    print(f'TT shape: {ms} -> {ns}')
    for idx in range(tt_cores):
        size = ns[:idx] + ms[idx + 1:] + (rs[idx] * ms[idx], rs[idx + 1] * ns[idx])
        core = torch.randn(*size, dtype=torch.float32)
        core.d_in = rs[idx] * ms[idx]
        core.d_out = ns[idx] * rs[idx + 1]
        core.in_dims = (-2,)
        core.out_dims = (-1,)
        cola_init(core, zero_init and idx == tt_cores - 1, init_method=init_method)
        cores.append(core)

    if normalize and learn_gamma:
        gamma0 = nn.Parameter(torch.tensor(1.))
        gamma1 = nn.Parameter(torch.tensor(1.))
        cores.append(gamma0)
        cores.append(gamma1)
    A = FasterBTT(cores, shapes, normalize)
    if permute:
        P_in = Permutation(torch.randperm(d_in), dtype=A.dtype)
        P_out = Permutation(torch.randperm(d_out), dtype=A.dtype)
        A = P_in @ A @ P_out
    return CoLALayer(A, bias=bias)

def build_btt_slow(d_in, d_out, tt_cores, tt_rank, permute=False, transpose=False, bias=True, zero_init=False, init_method='µP', **_):
    # tt_rank^2 should be much smaller than d_in and d_out
    ns = factorize(d_in, tt_cores)
    ms = factorize(d_out, tt_cores)
    print(f'TT shape: {ns} -> {ms}')
    cores = []
    for idx in range(tt_cores):
        n, m = ns[idx], ms[idx]
        rank_prev = 1 if idx == 0 else tt_rank
        rank_next = 1 if idx == tt_cores - 1 else tt_rank
        core = torch.rand(rank_next, rank_prev, *(ms[:idx] + [m, n] + ns[idx + 1:]))
        core.d_in = n * rank_prev
        core.d_out = m * rank_next
        core.in_dims = (1, -len(ns)+idx)
        cola_init(core, zero_init and idx == tt_cores - 1, init_method=init_method)
        cores.append(core)

    A = BTT(cores, transpose=transpose)
    if permute:
        P_in = Permutation(torch.randperm(d_in), dtype=A.dtype)
        P_out = Permutation(torch.randperm(d_out), dtype=A.dtype)
        A = P_in @ A @ P_out
    return CoLALayer(A, bias=bias)

def build_monarch(d_in, d_out, num_blocks=4, bias=True, zero_init=False, init_method='µP', **_):
    # Based on https://github.com/HazyResearch/fly/blob/master/src/models/layers/monarch_linear.py
    if num_blocks == -1:
        num_blocks = ceil(np.sqrt(d_in))
    in_blksz = int(ceil(d_in / num_blocks))
    out_blksz = int(ceil(d_out / num_blocks))
    d_int_ext = in_blksz * num_blocks
    d_out_ext = out_blksz * num_blocks

    if d_int_ext < d_out_ext:
        blkdiag1 = torch.empty(num_blocks, in_blksz, in_blksz)
        blkdiag2 = torch.empty(num_blocks, out_blksz, in_blksz)
    else:
        blkdiag1 = torch.empty(num_blocks, out_blksz, in_blksz)
        blkdiag2 = torch.empty(num_blocks, out_blksz, out_blksz)

    blkdiag1.d_in, blkdiag1.d_out = blkdiag1.shape[-1], blkdiag1.shape[-2]
    blkdiag2.d_in, blkdiag2.d_out = blkdiag2.shape[-1], blkdiag2.shape[-2]
    blkdiag1.in_dims, blkdiag1.out_dims = (-1,), (-2,)
    blkdiag2.in_dims, blkdiag2.out_dims = (-1,), (-2,)
    cola_init(blkdiag1, init_method=init_method)
    cola_init(blkdiag2, zero_init=zero_init, init_method=init_method)

    A = Monarch((blkdiag1, blkdiag2), shape=(d_in, d_out))
    return CoLALayer(A, bias=bias)

build_fns = {
    'low_rank': build_low_rank,
    'kron': build_tt,
    'tt': build_tt,
    'btt_slow': build_btt_slow,
    'btt': lambda *args, **kwargs: build_btt(*args, **kwargs),
    'btt_norm': lambda *args, **kwargs: build_btt(*args, normalize=True, **kwargs),
    'monarch': build_monarch,
    'dense': build_dense,
    'none': None,
}


def select_gpt_ffn_layers(name, layer_idx, num_layers):
    return 'mlp.c_fc' in name or 'mlp.c_proj' in name


def select_gpt_attn_layers(name, layer_idx, num_layers):
    return 'attn.c_attn' in name or 'attn.c_proj' in name


def select_ffn_layers(name, layer_idx, num_layers):
    return '.net.' in name


def select_attn_layers(name, layer_idx, num_layers):
    # return 'to_qkv' in name or 'to_out' in name
    keywords = ['to_q', 'to_k', 'to_v', 'to_qkv', 'to_out']
    return any(k in name for k in keywords)


layer_select_fns = {
    'all': lambda *_: True,
    'none': lambda *_: False,
    'all_but_last': lambda name, i, n: i < n - 1,
    'intermediate': lambda name, i, n: i > 0 and i < n - 1,
    'ffn': select_ffn_layers,
    'attn': select_attn_layers,
    'gpt_ffn': select_gpt_ffn_layers,
    'gpt_attn': select_gpt_attn_layers,
}


def zero_init_fn(weight, name):
    return hasattr(weight, 'zero_init') and weight.zero_init


def cola_parameterize(model_builder, base_config, lr, target_config=None, struct='none', layer_select_fn='all_but_last',
                      zero_init_fn=zero_init_fn, extra_lr_mult_fn=lambda p_name: 1, device='cuda', cola_kwargs={},
                      optim_kwargs={}, use_wrong_mult=False, init_method='µP'):
    """
    Create a model and its optimizer in an appropriate parameterization.
    Takes care of lr adjustment both for scaling up model size and for switching to CoLA layers.

    Usage:
        1. Regular μP: struct == 'none'
        2. Regular μP + dense layers initialized by CoLA: struct == 'dense' (e.g. specify custom zero inits)
        3. Regular μP + switching to arbitrary CoLA layers: struct == 'btt', 'tt', etc.

    Assumes:
    1. Classification head has been zero initialized.
    2. Every parameter tensor with more > 2 axes has been annotated with an attribute "fan_in_dims",
        a tuple of dimensions that are considered to be fan-in dimensions.
    3. If struct == 'none', init scale for the dense model is automatically in μP (often true with standard inits).
    4. layer_select_fn does not select the last layer.
    5. If struct != 'none', zero_init_fn selects the last linear layer in every residual block.

    Args:
        model_builder: function that builds the model
        base_config: kwargs for base model
        lr: learning rate for base model
        input_shape: shape of input model
        target_config: kwargs for scaled up model, same as base_config if not specified.
        init_mult: rescale initialization of all matrix-like parameters
        struct: structure of CoLA layers
        layer_select_fn: function that selects which layers to replace with CoLA layers
        zero_init_fn: function maps linear.weight and module name to whether to zero initialize
        extra_lr_mult_fn: function that maps parameter names to extra lr multipliers
        device: device to put model on
        cola_kwargs: kwargs for building CoLA layers
        optim_kwargs: kwargs for optimizer (AdamW)
    """
    base_model = deferred_init(model_builder, **base_config)
    base_param_shapes = [p.shape for p in base_model.parameters()]
    del base_model
    if target_config is None:
        target_config = base_config

    model = deferred_init(model_builder, **target_config)
    lr_mults = append_lr_mults_to_params(model, base_param_shapes=base_param_shapes)
    if struct != 'none':
        replace_layers_with_cola_layers(model, struct, cola_kwargs, layer_select_fn, zero_init_fn, use_wrong_mult, init_method)
    materialize_model(model, lr_mults)
    model.to(device)
    optimizer = adjust_lr_and_create_optimizer(model.named_parameters(), lr, extra_lr_mult_fn, optim_kwargs)
    return model, optimizer


def append_lr_mults_to_params(model, base_param_shapes):
    params, param_names, param_shapes = [], [], []
    for name, p in model.named_parameters():
        params.append(p)
        param_names.append(name)
        param_shapes.append(p.shape)
    # compute μP lr multipliers
    assert len(base_param_shapes) == len(param_shapes), 'Base model and target model have different number of param tensors'
    lr_mults = {}
    for base_shape, name, shape, p in zip(base_param_shapes, param_names, param_shapes, params):
        if base_shape == shape:
            mult = 1
        else:
            # special cases, e.g. word embeddings, positional embeddings
            if hasattr(p, 'fan_in_dims'):
                fan_in_dims = p.fan_in_dims
                d_in_base = base_shape[fan_in_dims]
                d_in = shape[fan_in_dims]
                if isinstance(d_in_base, tuple):
                    d_in_base = prod(d_in_base)
                    d_in = prod(d_in)
                mult = d_in_base / d_in
            # matrix like (d_out, d_in)
            elif len(base_shape) == 2:
                d_in_base = base_shape[1]
                d_in = shape[1]
                mult = d_in_base / d_in
            # vector like (d_out,), e.g. bias, layernorm gamma
            elif len(base_shape) == 1:
                mult = 1
            else:
                raise ValueError(
                    f'Non matrix or vector parameter {name} has shape {shape}, but does not have fan_in_dims attribute.')
        p.lr_mult = mult
        lr_mults[name] = mult
    return lr_mults


def replace_layers_with_cola_layers(model, struct, cola_kwargs, layer_select_fn, zero_init_fn, use_wrong_mult=False, init_method='µP'):
    if isinstance(struct, list):
        build_cola = build_fns[struct[0]]
    else:
        build_cola = build_fns[struct]
    if 'lm_head_struct' in cola_kwargs and cola_kwargs['lm_head_struct']:
        build_head = build_fns[cola_kwargs['lm_head_struct']]
        head_kwargs = cola_kwargs.copy()
        head_kwargs['tt_rank'] = cola_kwargs['lm_head_tt_rank']
        head_kwargs['rank_frac'] = cola_kwargs['lm_head_rank_frac']
    else:
        build_head = build_cola
        head_kwargs = cola_kwargs
    if isinstance(layer_select_fn, str):
        layer_select_fn = layer_select_fns[layer_select_fn]
    assert callable(layer_select_fn), f'layer_select_fn must be callable, got {layer_select_fn}'
    # Create a list of all linear layers and their names
    linear_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    num_layers = len(linear_layers)
    for layer_idx, (name, module) in enumerate(linear_layers):
        if layer_select_fn(name, layer_idx, num_layers):
            d_in, d_out = module.in_features, module.out_features
            bias = module.bias is not None
            zero_init = zero_init_fn(module.weight, name)
            if zero_init:
                print(f'Zero init: {name}')
            assert hasattr(module.weight, 'lr_mult'), 'Weights in linear layer must have lr_mult attribute'
            dense_lr_mult = module.weight.lr_mult

            learn_gamma = True
            if 'do_qk_ln' in cola_kwargs and cola_kwargs['do_qk_ln']:
                # when doing qk LN, no need to learn gamma for W_Q, W_K
                qk_layer_names = ['c_attn_q', 'c_attn_k', 'to_q', 'to_k']
                learn_gamma = not any(qk_layer_name in name for qk_layer_name in qk_layer_names)
                if not learn_gamma:
                    print(f'Not learning gamma for {name} due to qk ln')
            if name == 'lm_head':
                cola_layer = build_head(d_in, d_out, bias=bias, zero_init=zero_init, **head_kwargs,
                                        learn_gamma=learn_gamma)  # cola specific lr mult are attached
            elif isinstance(struct, list):
                build_cola = build_fns[struct[layer_idx]]
                cola_layer = build_cola(d_in, d_out, bias=bias, zero_init=zero_init, cola_kwargs=cola_kwargs[layer_idx],
                                        learn_gamma=learn_gamma, init_method=init_method)
            else:
                cola_layer = build_cola(d_in, d_out, bias=bias, zero_init=zero_init, **cola_kwargs,
                                        learn_gamma=learn_gamma, init_method=init_method)  # cola specific lr mult are attached

            for p in cola_layer.op_params:
                if use_wrong_mult:
                    p.lr_mult = dense_lr_mult
                else:
                    if p.dim() == 0:
                        p.lr_mult = 1  # scalar have O(1) lr
                    else:
                        p.lr_mult = p.lr_mult * dense_lr_mult  # final cola mult = cola mult * dense mult
            # Split the name to get parent module and attribute name
            name_split = name.rsplit('.', 1)
            # If it's a top-level module
            if len(name_split) == 1:
                setattr(model, name_split[0], cola_layer)
            else:
                parent_name, attr_name = name_split
                parent_module = reduce(getattr, parent_name.split('.'), model)
                setattr(parent_module, attr_name, cola_layer)


def materialize_model(model, lr_mults):
    materialize_module(model)
    # materialization gets rid of lr_mults of those tensors, so we need to add them back
    for name, p in model.named_parameters():
        if not hasattr(p, 'lr_mult'):
            if hasattr(p, 'override_lr'):
                pass
            else:
                p.lr_mult = lr_mults[name]

def adjust_lr_and_create_optimizer(named_parameters, lr, extra_lr_mult_fn, optim_kwargs):
    wd = optim_kwargs['weight_decay'] if 'weight_decay' in optim_kwargs else 0.0
    device_type = optim_kwargs.pop("device_type") if "device_type" in optim_kwargs else "cuda"
    param_groups = []
    for name, param in named_parameters:
        if hasattr(param, 'override_lr'):
            param.lr_mult = param.override_lr / lr
        assert hasattr(param, 'lr_mult'), f'lr_mult not found for {name}'
        mult = param.lr_mult
        extra_mult = extra_lr_mult_fn(name)
        if extra_mult != 1:
            print(f'{name}: {extra_mult} * {mult}')
            mult *= extra_mult
        else:
            print(f'{name}: {mult}')
        if param.dim() >= 2 and param.requires_grad:
            adjusted_wd = wd
        else:
            print(f'No wd for {name}')
            adjusted_wd = 0.0
        adjusted_lr = lr * mult
        param_groups.append({'params': param, 'lr': adjusted_lr, 'weight_decay': adjusted_wd})
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    opt_name = optim_kwargs.pop("opt_name") if "opt_name" in optim_kwargs else "AdamW"
    if opt_name == "AdamW":
        optimizer = AdamW(param_groups, **optim_kwargs, **extra_args)
    elif opt_name == "sgd":
        if "fused" in extra_args.keys():
            extra_args.pop("fused")
        optimizer = SGD(param_groups, **optim_kwargs, **extra_args)
    else:
        optimizer = Adam(param_groups, **optim_kwargs, **extra_args)
    return optimizer


def get_model_summary_and_flops(model, fake_input):
    print('Model:')
    stats = summary(model, input_data=fake_input)
    cola_params = stats.trainable_params
    print(f'Params: {cola_params / 1e6:.2f}M')
    cola_flops = FlopCountAnalysis(model, fake_input).set_op_handle(**custom_ops).total()
    print(f'FLOPs: {cola_flops / 1e6:.2f}M')
    print('=' * 90)
    info = {'cola_params': cola_params, 'cola_flops': cola_flops}

    return info


def btt_flop_count(inputs, outputs):
    x, W1, W2 = inputs
    batch_size = get_shape(x)[0]
    params = get_numel(W1) + get_numel(W2)
    return batch_size * params


def scaled_dot_product_attention_flop_count(inputs, outputs):
    output_shape = get_shape(outputs[0])  # ([batch dims], seq_len, head_dim)
    B = prod(output_shape[:-2])  # batch suze
    L, D = output_shape[-2], output_shape[-1]  # (seq_len, head_dim)
    qk_flops = B * (L * D * L)  # (L, D) @ (D, L)
    v_flops = B * (L * L * D)  # (L, L) @ (L, D)
    return qk_flops + v_flops


def get_shape(x):
    return x.type().sizes()


def get_numel(x):
    return prod(get_shape(x))


def normalized_rms(x):
    d_in, d_out = x.d_in, x.d_out
    rms = torch.sqrt(torch.mean(x**2) + 1e-8).item()
    expected = np.sqrt(min(d_in, d_out) * d_out / (d_out * d_in * d_in))
    return rms / expected

def rms(x, dim, eps=1e-16):
    return torch.sqrt(torch.mean(x**2, dim) + eps)

def rms_norm(x, dim):
    return x / rms(x, dim).unsqueeze(dim)

custom_ops = {
    'prim::PythonOp.BlockdiagButterflyMultiply': btt_flop_count,
    'prim::PythonOp.BlockTeTr': btt_flop_count,
    'aten::scaled_dot_product_attention': scaled_dot_product_attention_flop_count
}
