import wandb
from nn.cola_nn import cola_parameterize, get_model_summary_and_flops
import nn
from fastargs.validation import And, OneOf
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs import get_current_config
from math import prod
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4
import json
import time
import os
from tqdm import tqdm
import numpy as np
import torchmetrics
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from contextlib import nullcontext
import torch.distributed as dist

from timm.data import create_dataset, create_loader, FastCollateMixup, AugMixDataset
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

Section('model', 'model details').params(
    arch=Param(str, default='ViT'),
    base_width=Param(int, default=256),
    base_depth=Param(int, default=6),
    base_heads=Param(int, default=4),
    base_dim_head=Param(int, default=64),
    base_ffn_expansion=Param(int, default=4),
    width=Param(int, default=-1),
    depth=Param(int, default=-1),
    heads=Param(int, default=-1),
    dim_head=Param(int, default=-1),
    ffn_expansion=Param(int, default=4),
    patch_size=Param(int, default=16),
    in_channels=Param(int, default=3),
    resolution=Param(int, default=224),
    fixup=Param(int, default=0),
    dropout=Param(float, default=0.),
)

Section('cola', 'cola details').params(
    struct=Param(str, default='none'),
    layers=Param(str, default='all_but_last'),
    tt_rank=Param(int, default=1),
    tt_dim=Param(int, default=2),
    num_blocks=Param(int, default=4),
)

Section('data', 'data related stuff').params(data_dir=Param(str, 'data location', required=True),
                                             num_workers=Param(int, 'The number of workers', required=True),
                                             in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True))

Section('logging', 'how to log stuff').params(folder=Param(str, 'log location',
                                                           required=True), log_level=Param(int, '0 if only at end 1 otherwise',
                                                                                           default=1),
                                              log_freq=Param(int, 'how often to log',
                                                             default=100), use_wandb=Param(int, 'use wandb?', default=1),
                                              wandb_project=Param(str, 'wandb project name', default='imagenet'),
                                              wandb_group=Param(str, 'wandb group name', default=''))

Section('training', 'training hyper param stuff').params(
    grad_accum_steps=Param(int, 'gradient accumulation steps', default=1),
    schedule=Param(OneOf(['none', 'cosine']), default='none'),
    lr=Param(float, 'lr', default=3e-4),
    input_lr_mult=Param(float, default=1),
    init_mult_1=Param(float, default=1),
    init_mult_2=Param(float, default=1),
    lr_mult_1=Param(float, default=1),
    lr_mult_2=Param(float, default=1),
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['adamw'])), 'The optimizer', default='adamw'),
    weight_decay=Param(float, 'weight decay', default=0),
    epochs=Param(int, 'number of epochs', default=30),
    warmup_epochs=Param(int, 'number of warmup epochs', default=0),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.),
    mixup=Param(float, 'mixup parameter', default=0.),
    distributed=Param(int, 'is distributed?', default=0),
    mixed_prec=Param(int, 'mixed precision?', default=1),
    clip_grad=Param(float, 'clip grad', default=0),
)

Section('validation',
        'Validation parameters stuff').params(lr_tta=Param(int, 'should do lr flipping/avging at test time', default=0))

Section('dist', 'distributed training options').params(world_size=Param(int, 'number gpus', default=1),
                                                       address=Param(str, 'address', default='localhost'),
                                                       port=Param(str, 'port', default='12355'))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CROP_RATIO = 224 / 256


@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4, 1, 0]
    return np.interp([epoch], xs, ys)[0]


@param('training.warmup_epochs')
def get_constant_mult(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 1


@param('training.warmup_epochs')
@param('training.epochs')
def get_cosine_mult(epoch, warmup_epochs, epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))


class ImageNetTrainer:
    @param('training.distributed')
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader, self.val_loader, self.loss, self.val_loss = self.create_loader_and_loss()
        self.example_imgs = next(iter(self.val_loader))[0]  # for tracking features
        self.model, self.scaler, self.optimizer, self.info = self.create_model_and_scaler()
        self.initialize_logger()
        self.prev_hs = None  # prev hidden states for feature tracking
        self.mixed_prec = self.all_params['training.mixed_prec']

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('training.schedule')
    def get_lr_sched_mult(self, epoch, schedule):
        lr_schedules = {'none': get_constant_mult, 'cosine': get_cosine_mult}

        return lr_schedules[schedule](epoch)

    # resolution tools
    @param('model.resolution')
    def get_resolution(self, resolution):
        return resolution

    @param('data.data_dir')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.mixup')
    @param('training.label_smoothing')
    @param('training.distributed')
    @param('training.mixed_prec')
    @param('data.in_memory')
    def create_loader_and_loss(self, data_dir, num_workers, batch_size, mixup, label_smoothing, distributed, mixed_prec,
                               in_memory):
        device = ch.device(f'cuda:{self.gpu}')
        dataset_train = create_dataset(
            '',
            root=data_dir,
            split='train',
            is_training=True,
            class_map='',
            download=False,
            batch_size=batch_size,
            seed=42,
            repeats=0,
            input_img_mode='RGB',
            input_key=None,
            target_key=None,
            num_samples=None,
        )

        dataset_eval = create_dataset(
            '',
            root=data_dir,
            split='validation',
            is_training=False,
            class_map='',
            download=False,
            batch_size=batch_size,
            input_img_mode='RGB',
            input_key=None,
            target_key=None,
            num_samples=None,
        )

        prefetcher = True
        # setup mixup / cutmix
        collate_fn = None
        mixup_active = mixup > 0
        num_aug_splits = 0
        if mixup_active:
            mixup_args = dict(mixup_alpha=mixup, cutmix_alpha=0, cutmix_minmax=None, prob=1, switch_prob=0.5, mode='batch',
                              label_smoothing=label_smoothing, num_classes=1000)
            if prefetcher:
                assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                raise NotImplementedError("Mixup not supported without prefetcher because I'm lazy")

        # wrap dataset in AugMix helper
        if num_aug_splits > 1:
            dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

        # create data loaders w/ augmentation pipeiine
        train_interpolation = 'random'
        no_aug = False
        loader_train = create_loader(
            dataset_train,
            input_size=(3, 224, 224),
            batch_size=batch_size,
            is_training=True,
            no_aug=no_aug,
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            re_split=False,
            scale=[0.08, 1.0],
            ratio=[0.75, 1.3333333333333333],
            hflip=0.5,
            vflip=0,
            color_jitter=0.4,
            color_jitter_prob=None,
            grayscale_prob=None,
            gaussian_blur_prob=None,
            auto_augment='rand-m9-mstd0.5-inc1',
            num_aug_repeats=0,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            num_workers=num_workers,
            distributed=distributed,
            collate_fn=collate_fn,
            pin_memory=False,
            device=device,
            use_prefetcher=prefetcher,
            use_multi_epochs_loader=False,
            worker_seeding='all',
        )

        loader_eval = None
        val_split = 'validation'
        if val_split:
            eval_workers = num_workers
            loader_eval = create_loader(
                dataset_eval,
                input_size=(3, 224, 224),
                batch_size=batch_size,
                is_training=False,
                interpolation='bicubic',
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                num_workers=eval_workers,
                distributed=distributed,
                crop_pct=DEFAULT_CROP_RATIO,
                pin_memory=False,
                device=device,
                use_prefetcher=prefetcher,
            )

        if mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            train_loss_fn = SoftTargetCrossEntropy()
        elif label_smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            train_loss_fn = ch.nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(device=device)
        validate_loss_fn = ch.nn.CrossEntropyLoss().to(device=device)

        return loader_train, loader_eval, train_loss_fn, validate_loss_fn

    @param('training.epochs')
    @param('logging.log_level')
    def train(self, epochs, log_level):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {'train_loss': train_loss, 'train_acc': train_acc, 'epoch': epoch}

                self.eval_and_log(extra_dict)
                if self.gpu == 0:
                    state = {
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    ch.save(state, self.log_folder / 'checkpoint.pt')

        self.eval_and_log({'epoch': epoch})
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')
            with open(self.log_file, 'a') as f:
                f.write('Finished training!\n')

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(
                dict(
                    {
                        'current_lr': self.optimizer.param_groups[0]['lr'],
                        'val_loss': stats['loss'],
                        'val_acc': stats['top_1'],
                        'top_5': stats['top_5'],
                        'val_time': val_time
                    }, **extra_dict))
            # open log file and write val loss and acc
            with open(self.log_file, 'a') as f:
                epoch = extra_dict['epoch']
                log_str = f'E {epoch} | A {stats["top_1"]:.2f} | L {stats["loss"]:.4f}'
                log_str += f' | At {extra_dict["train_acc"]:.2f} | Lt {extra_dict["train_loss"]:.4f}'
                # expected time to finish
                eta = (time.time() - self.start_time) / (epoch + 1) * (self.all_params['training.epochs'] - epoch - 1)
                log_str += f' | ETA: {eta / 3600:.1f} hours\n'
                f.write(log_str)
        return stats

    @param('training.distributed')
    @param('model.arch')
    @param('model.base_width')
    @param('model.base_depth')
    @param('model.base_heads')
    @param('model.base_dim_head')
    @param('model.base_ffn_expansion')
    @param('model.width')
    @param('model.depth')
    @param('model.heads')
    @param('model.dim_head')
    @param('model.ffn_expansion')
    @param('model.patch_size')
    @param('model.in_channels')
    @param('model.resolution')
    @param('model.fixup')
    @param('model.dropout')
    @param('cola.struct')
    @param('cola.layers')
    @param('cola.tt_rank')
    @param('cola.tt_dim')
    @param('cola.num_blocks')
    @param('training.lr')
    @param('training.weight_decay')
    @param('training.input_lr_mult')
    @param('training.init_mult_1')
    @param('training.init_mult_2')
    @param('training.lr_mult_1')
    @param('training.lr_mult_2')
    @param('training.optimizer')
    @param('training.label_smoothing')
    def create_model_and_scaler(self, distributed, arch, base_width, base_depth, base_heads, base_dim_head, base_ffn_expansion,
                                width, depth, heads, dim_head, ffn_expansion, patch_size, in_channels, resolution, fixup, dropout,
                                struct, layers, tt_rank, tt_dim, num_blocks, lr, weight_decay, input_lr_mult, init_mult_1,
                                init_mult_2, lr_mult_1, lr_mult_2, optimizer, label_smoothing):
        assert optimizer == 'adamw', 'Only adamw supported'
        scaler = GradScaler()
        model_builder = getattr(nn, arch)
        input_shape = (1, in_channels, resolution, resolution)
        base_config = dict(dim_in=prod(input_shape), dim_out=1000, depth=base_depth, width=base_width, heads=base_heads,
                           dim_head=base_dim_head, ffn_expansion=base_ffn_expansion, patch_size=patch_size,
                           in_channels=in_channels, image_size=resolution, fixup=fixup, dropout=dropout)

        depth = base_depth if depth == -1 else depth
        width = base_width if width == -1 else width
        if heads == -1 and dim_head == -1:
            heads = base_heads if heads == -1 else heads
            dim_head = base_dim_head if dim_head == -1 else dim_head
        elif heads != -1:
            dim_head = width // heads
        elif dim_head != -1:
            heads = width // dim_head
        self.depth, self.width, self.heads, self.dim_head = depth, width, heads, dim_head
        target_config = dict(dim_in=prod(input_shape), dim_out=1000, depth=depth, width=width, heads=heads, dim_head=dim_head,
                             ffn_expansion=ffn_expansion, patch_size=patch_size, image_size=resolution, in_channels=in_channels,
                             fixup=fixup, dropout=dropout)
        # update width, depth, etc. for logging
        self.target_config = target_config

        def extra_lr_mult_fn(param_name):
            if 'to_patch_embedding' in param_name or 'input_layer' in param_name:
                return input_lr_mult
            elif 'matrix_params.0' in param_name:
                return lr_mult_1
            elif 'matrix_params.1' in param_name:
                return lr_mult_2
            else:
                return 1

        def extra_init_mult_fn(param_name):
            if 'matrix_params.0' in param_name:
                print(f'scaling {param_name} std by {init_mult_1}')
                return init_mult_1
            elif 'matrix_params.1' in param_name:
                print(f'scaling {param_name} std by {init_mult_2}')
                return init_mult_2
            else:
                return 1

        def zero_init_fn(weight, name):
            return hasattr(weight, 'zero_init') and weight.zero_init

        cola_kwargs = dict(tt_dim=tt_dim, tt_rank=tt_rank, num_blocks=num_blocks, do_qk_ln=True)
        optim_kwargs = dict(weight_decay=weight_decay, betas=(0.9, 0.95))
        model, optimizer = cola_parameterize(model_builder, base_config=base_config, lr=lr, target_config=target_config,
                                             struct=struct, layer_select_fn=layers, zero_init_fn=zero_init_fn,
                                             extra_lr_mult_fn=extra_lr_mult_fn, device='cuda', cola_kwargs=cola_kwargs,
                                             optim_kwargs=optim_kwargs)
        fake_input = ch.randn(*input_shape).to('cuda')
        info = get_model_summary_and_flops(model, fake_input)
        # model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)
        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            if hasattr(model.module, 'get_features'):
                model.get_features = model.module.get_features
                model.clear_features = model.module.clear_features
        return model, scaler, optimizer, info

    @param('logging.log_level')
    @param('logging.log_freq')
    @param('training.clip_grad')
    @param('training.mixup')
    @param('training.grad_accum_steps')
    def train_loop(self, epoch, log_level, log_freq, clip_grad, mixup, grad_accum_steps):
        model = self.model
        model.train()
        losses = []
        accs = []

        raw_lrs = [group['lr'] for group in self.optimizer.param_groups]
        lr_sched_mult_start, lr_sched_mult_end = self.get_lr_sched_mult(epoch), self.get_lr_sched_mult(epoch + 1)
        iters = len(self.train_loader)
        lr_sched_mults = np.interp(np.arange(iters), [0, iters], [lr_sched_mult_start, lr_sched_mult_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            # Training start
            for param_group, raw_lr in zip(self.optimizer.param_groups, raw_lrs):
                param_group['lr'] = raw_lr * lr_sched_mults[ix]

            with autocast() if self.mixed_prec else nullcontext():
                output = self.model(images)
                loss_train = self.loss(output, target)
                loss_train = loss_train / grad_accum_steps

            self.scaler.scale(loss_train).backward()

            if (ix + 1) % grad_accum_steps == 0:
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                if clip_grad > 0:
                    ch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                # optimizer's gradients are already unscaled, so self.scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            losses.append(loss_train.detach() * grad_accum_steps)  # Multiply by grad_accum_steps to get the actual loss value
            if mixup > 0:
                accs.append((output.argmax(dim=-1) == target.argmax(dim=-1)).float().mean().detach() * 100)
            else:
                accs.append((output.argmax(dim=-1) == target).float().mean().detach() * 100)
            # Training end

            # Logging start
            if log_level > 0 and ix % log_freq == 0:
                feature_metrics = self.get_features_metrics(self.example_imgs)
                cur_lr = self.optimizer.param_groups[0]['lr']

                msg = f'E {epoch} | L {loss_train:.2f} | A {accs[-1]:.1f}'
                iterator.set_description(msg)

                self.log(
                    dict(
                        {
                            'train_loss': ch.tensor(losses).mean().item(),
                            'train_acc': ch.tensor(accs).mean().item(),
                            'lr': cur_lr,
                        }, **feature_metrics))
            # Logging end
        # Reset lr
        for param_group, raw_lr in zip(self.optimizer.param_groups, raw_lrs):
            param_group['lr'] = raw_lr
        return ch.tensor(losses).mean().item(), ch.tensor(accs).mean().item()

    def get_features_metrics(self, val_imgs):
        metrics = {}
        if hasattr(self.model, 'get_features'):
            self.model.clear_features()
            with autocast() if self.mixed_prec else nullcontext():
                self.model.eval()  # features only saved in eval mode
                self.model(val_imgs)
                self.model.train()
            hs = self.model.get_features()
            hs = [ch.cat(h.buffer, dim=0) for h in hs]  # list of tensors
            if self.prev_hs is None:
                self.prev_hs = hs
            dhs = [hs[i] - self.prev_hs[i] for i in range(len(hs))]
            h_norm = [rms(h) for h in hs]  # should be O(1)
            h_std = [ch.std(h) for h in hs]
            dh_norm = [rms(dh) for dh in dhs]  # should be O(1)
            self.prev_hs = hs
            metrics = {}
            for i in range(len(h_norm)):
                metrics[f'h_{i}'] = h_norm[i].item()
                metrics[f'std_{i}'] = h_std[i].item()
                metrics[f'dh_{i}'] = dh_norm[i].item()
            # go through all params
            for name, p in self.model.named_parameters():
                if hasattr(p, 'rms'):
                    metrics[f'rms/{name}'] = p.rms
                if hasattr(p, 'x'):
                    metrics[f'in/{name}'] = p.x
                if hasattr(p, 'out'):
                    metrics[f'out/{name}'] = p.out
                if hasattr(p, 'scale'):
                    metrics[f'scale/{name}'] = p.scale
        return metrics

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast() if self.mixed_prec else nullcontext():
                for images, target in tqdm(self.val_loader):
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)

                    loss_val = self.val_loss(output, target)  # no label smoothing
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    @param('logging.use_wandb')
    @param('logging.wandb_project')
    @param('logging.wandb_group')
    def initialize_logger(self, folder, use_wandb, wandb_project, wandb_group):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(self.gpu) * 100,
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(self.gpu) * 100,
            'loss': MeanScalarMetric().to(self.gpu)
        }

        if self.gpu == 0:
            folder = (Path(folder) / self.runname() / str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.log_file = folder / 'log.txt'
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {'.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()}

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)
            config_dict = {k.split('.')[-1]: v for k, v in params.items()}
            config_dict.update(self.target_config)
            config_dict.update(self.info)
            config_dict['model'] = config_dict['arch']
            config_dict['ckpt_dir'] = str(folder)
            config_dict['runname'] = self.runname()
            # write config into log file one variable per line
            with open(self.log_file, 'a') as f:
                for k, v in config_dict.items():
                    f.write(f'{k}: {v}\n')
            if use_wandb:
                wandb.init(project=wandb_project, name=self.runname(), group=wandb_group if wandb_group else None,
                           config=config_dict)

    @param('logging.use_wandb')
    def log(self, content, use_wandb):
        if self.gpu != 0:
            return
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({'timestamp': cur_time, 'relative_time': cur_time - self.start_time, **content}) + '\n')
            fd.flush()
        if use_wandb:
            wandb.log(content)

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            print(f'Launching distributed training with {world_size} gpus')
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

    @param('model.arch')
    @param('model.patch_size')
    @param('cola.struct')
    @param('cola.layers')
    @param('cola.tt_dim')
    @param('cola.tt_rank')
    @param('training.lr')
    @param('training.weight_decay')
    @param('training.batch_size')
    def runname(self, arch, patch_size, struct, layers, tt_dim, tt_rank, lr, weight_decay, batch_size):
        name = arch + f'-{patch_size}'
        name += f'_d{self.depth}w{self.width}h{self.heads}dh{self.dim_head}'
        if struct != 'none':
            name += f'_{struct}_{layers}_tt_{tt_dim}_{tt_rank}'
        name += f'_lr{lr}wd{weight_decay}bs{batch_size}'
        return name


# Utils


class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count


def rms(x, eps=1e-8):
    x = x.float()  # to avoid nan
    return (ch.mean(x**2) + eps).sqrt()


# Running


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()


if __name__ == "__main__":
    make_config()
    ImageNetTrainer.launch_from_args()