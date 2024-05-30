import os
import time
import json
from math import prod
import wandb
import torch
from torch.nn import CrossEntropyLoss
from model import set_seed
from nn.cola_nn import cola_parameterize, get_model_summary_and_flops
import nn
from tqdm import tqdm
from scaling_mlps.data_utils import data_stats
from scaling_mlps.data_utils.dataloader import get_loader
from scaling_mlps.utils.config import config_to_name
from scaling_mlps.utils.metrics import topk_acc, real_acc, AverageMeter
from scaling_mlps.utils.optimizer import get_scheduler
from scaling_mlps.utils.parsers import get_training_parser
from einops import rearrange


def train(model, opt, scheduler, loss_fn, epoch, train_loader, args):
    start = time.time()
    model.train()

    total_acc, total_top5 = AverageMeter(), AverageMeter()
    total_loss = AverageMeter()
    total_aux_loss = AverageMeter()

    for step, (ims, targs) in enumerate(train_loader):
        preds = model(ims)

        if args.mixup > 0:
            targs_perm = targs[:, 1].long()
            weight = targs[0, 2].squeeze()
            targs = targs[:, 0].long()
            if weight != -1:
                loss = loss_fn(preds, targs) * weight + loss_fn(preds, targs_perm) * (1 - weight)
            else:
                loss = loss_fn(preds, targs)
                targs_perm = None
        else:
            if args.ar_modeling:
                targs = rearrange(ims, 'b c h w -> (b h w c)')
                preds = preds[:, :-1].reshape(-1, preds.shape[-1])  # nothing to predict after the last pixel
            loss = loss_fn(preds, targs)
            targs_perm = None

        # load balancing loss
        aux_losses = []
        spec_penalties = []
        for name, module in model.named_modules():
            if 'moe_gate' in name:
                aux_losses.append(module.load_balancing_loss)
            if hasattr(module, 'natural_norm'):
                spec_penalties.append(module.natural_norm**2)
        aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else 0
        spec_penalty = sum(spec_penalties) / len(spec_penalties) if spec_penalties else 0
        loss += aux_loss * args.aux_loss_weight
        loss += spec_penalty * args.spec_penalty_weight

        acc, top5 = topk_acc(preds, targs, targs_perm, k=5, avg=True)
        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])

        loss = loss / args.accum_steps
        loss.backward()

        if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            opt.zero_grad()

        total_loss.update(loss.item() * args.accum_steps, ims.shape[0])
        total_aux_loss.update(aux_loss * args.accum_steps, ims.shape[0])

    end = time.time()

    scheduler.step()

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
        total_loss.get_avg(percentage=False),
        total_aux_loss.get_avg(percentage=False),
        end - start,
    )


@torch.no_grad()
def test(model, loader, loss_fn, args):
    start = time.time()
    model.eval()
    total_acc, total_top5, total_loss = AverageMeter(), AverageMeter(), AverageMeter()

    for ims, targs in loader:
        preds = model(ims)
        if args.ar_modeling:
            targs = rearrange(ims, 'b c h w -> (b h w c)')
            preds = preds[:, :-1].reshape(-1, preds.shape[-1])  # nothing to predict after the last pixel
        if args.dataset != 'imagenet_real':
            acc, top5 = topk_acc(preds, targs, k=5, avg=True)
            loss = loss_fn(preds, targs).item()
        else:
            acc = real_acc(preds, targs, k=5, avg=True)
            top5 = 0
            loss = 0

        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])
        total_loss.update(loss)

    end = time.time()

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
        total_loss.get_avg(percentage=False),
        end - start,
    )


def main(args):
    set_seed(args.seed)
    # Use mixed precision matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_shape = (1, 3, args.crop_resolution, args.crop_resolution)
    model_builder = getattr(nn, args.model)
    base_ffn_expansion = 4  # equivalent to specifying a constant LR multuplier in μP. 4 works well for ViT.
    base_config = dict(dim_in=prod(input_shape), dim_out=args.num_classes, depth=args.depth, width=args.width,
                       num_ffn_experts=args.num_ffn_experts, ffn_expansion=base_ffn_expansion, patch_size=args.patch_size,
                       in_channels=args.in_channels, shuffle_pixels=args.shuffle_pixels, heads=args.heads, dim_head=args.dim_head,
                       attn_mult=args.attn_mult, output_mult=args.output_mult, emb_mult=args.emb_mult, layer_norm=args.layer_norm)
    # target config
    target_config = base_config.copy()
    args.width = int(args.width * args.scale_factor)  # we update args.width to have it logged in wandb
    target_config['width'] = args.width
    target_config['ffn_expansion'] = args.ffn_expansion

    # additional LR multipliers
    def extra_lr_mult_fn(param_name):
        if 'to_patch_embedding' in param_name or 'input_layer' in param_name:
            return args.input_lr_mult
        elif 'op_params.0' in param_name:
            print(f'scaling {param_name} LR by {args.lr_mult_1}')
            return args.lr_mult_1
        elif 'op_params.1' in param_name:
            print(f'scaling {param_name} LR by {args.lr_mult_2}')
            return args.lr_mult_2
        else:
            return 1

    def extra_init_mult_fn(param_name):
        if 'op_params.0' in param_name:
            print(f'scaling {param_name} std by {args.init_mult_1}')
            return args.init_mult_1
        elif 'op_params.1' in param_name:
            print(f'scaling {param_name} std by {args.init_mult_2}')
            return args.init_mult_2
        else:
            return 1

    def zero_init_fn(weight, name):
        return hasattr(weight, 'zero_init') and weight.zero_init

    # CoLA structure
    struct = args.struct
    fact_cls = None
    if struct.startswith("einsum"):
        fact_cls = select_factorizer(name=args.fact)
        fact_cls = fact_cls(cores_n=args.cores_n, int_pow=args.int_pow)
        fact_cls.sample(expr=args.expr)
        base_config["fact_cls"] = fact_cls
        target_config["fact_cls"] = fact_cls
    cola_kwargs = dict(tt_cores=args.tt_cores, tt_rank=args.tt_rank, num_blocks=args.num_blocks, rank_frac=args.rank_frac,
                       fact_cls=fact_cls, expr=args.expr, init_type=args.init_type, do_sgd_lr=args.optimizer == "sgd")
    # initialize scaled up model with some linear layers replaced by cola layers,
    # and create optimizer with appropriately scaled learning rates
    if args.use_wrong_mult:
        print("#### WARNING: using wrong mult ####")
    optim_kwargs = {"opt_name": args.optimizer}
    model, opt = cola_parameterize(model_builder, base_config, args.lr, target_config=target_config, struct=struct,
                                   layer_select_fn=args.layers, zero_init_fn=zero_init_fn, extra_lr_mult_fn=extra_lr_mult_fn,
                                   device=device, cola_kwargs=cola_kwargs, use_wrong_mult=args.use_wrong_mult, init_method=args.init_method,
                                   optim_kwargs=optim_kwargs)
    fake_input = torch.zeros(*input_shape).to('cuda')
    if args.ar_modeling:
        fake_input = fake_input.long()
    info = get_model_summary_and_flops(model, fake_input)
    # if struct == "einsum":
    #     info["cola_flops"] += fact_cls.flops
    #     print(fact_cls.layers)

    scheduler = get_scheduler(opt, args.scheduler, **args.__dict__)
    loss_fn = CrossEntropyLoss(label_smoothing=args.smooth)

    # Create unique identifier
    run_name = config_to_name(args)
    path = os.path.join(args.checkpoint_folder, run_name)

    # Create folder to store the checkpoints
    if not os.path.exists(path):
        os.makedirs(path)
        with open(path + '/config.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Get the dataloaders
    local_batch_size = args.batch_size // args.accum_steps

    train_loader = get_loader(args.dataset, bs=local_batch_size, mode="train", augment=args.augment, dev=device,
                              num_samples=args.n_train, mixup=args.mixup, data_path=args.data_path,
                              data_resolution=args.resolution, crop_resolution=args.crop_resolution, ar_modeling=args.ar_modeling)

    test_loader = get_loader(args.dataset, bs=local_batch_size, mode="test", augment=False, dev=device, data_path=args.data_path,
                             data_resolution=args.resolution, crop_resolution=args.crop_resolution, ar_modeling=args.ar_modeling)

    if args.wandb:
        config = args.__dict__
        config.update(info)
        if struct.startswith("ein_expr"):
            config.update({"expr0": args.expr})
        if struct.startswith("einsum"):
            config.update(fact_cls.log_data())
            exprs = fact_cls.get_unique_ein_expr()
            formated_combined = {f"expr{idx}": f"{key}({val:d})" for idx, (key, val) in enumerate(exprs.items())}
            config.update(formated_combined)
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            tags=["pretrain", args.dataset],
        )

    compute_per_epoch = info['cola_flops'] * len(train_loader) * args.batch_size

    prev_hs = None
    recent_train_accs = []
    recent_train_losses = []
    start_ep = 0
    for ep in (pb := tqdm(range(start_ep, args.epochs))):
        calc_stats = ep == start_ep or ep == args.epochs - 1 or (ep + 1) % args.calculate_stats == 0

        current_compute = compute_per_epoch * ep
        if ep == 0:  # skip first epoch
            train_acc, train_top5, train_loss, aux_loss, train_time = 0, 0, 0, 0, 0
        else:
            train_acc, train_top5, train_loss, aux_loss, train_time = train(model, opt, scheduler, loss_fn, ep, train_loader,
                                                                            args)
        if len(recent_train_accs) == 10:
            recent_train_accs.pop(0)
        recent_train_accs.append(train_acc)
        if len(recent_train_losses) == 10:
            recent_train_losses.pop(0)
        recent_train_losses.append(train_loss)

        if calc_stats:
            # model.hs = [[] for _ in range(len(model.hs))]  # clear the list
            model.clear_features()
            test_acc, test_top5, test_loss, test_time = test(model, test_loader, loss_fn, args)
            # get features on test set
            # hs = model.hs  # list of lists of tensors
            hs = model.get_features()
            hs = [torch.cat(h.buffer, dim=0) for h in hs]  # list of tensors
            if prev_hs is None:
                prev_hs = hs
            dhs = [hs[i] - prev_hs[i] for i in range(len(hs))]
            h_norm = [torch.norm(h, dim=1).mean() / h.shape[1]**0.5 for h in hs]  # should be O(1)
            dh_norm = [torch.norm(dh, dim=1).mean() / dh.shape[1]**0.5 for dh in dhs]  # should be O(1)
            prev_hs = hs

            if args.wandb:
                logs = {
                    "epoch": ep,
                    "train_acc": train_acc,
                    "train_acc_avg": sum(recent_train_accs) / len(recent_train_accs),
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "train_loss": train_loss,
                    "train_loss_avg": sum(recent_train_losses) / len(recent_train_losses),
                    "aux_loss": aux_loss,
                    "current_compute": current_compute,
                    "Inference time": test_time,
                }
                for i in range(len(h_norm)):
                    logs[f'h_{i}'] = h_norm[i].item()
                    logs[f'dh_{i}'] = dh_norm[i].item()
                # go through all params
                for name, p in model.named_parameters():
                    # if hasattr(p, 'rms'):
                    #     logs[f'rms/{name}'] = p.rms
                    if 'top_singular_vec' in name:
                        logs[f'v_rms/{name}'] = torch.sqrt((p**2).mean()).item()
                    if hasattr(p, 'scale'):
                        logs[f'scale/{name}'] = p.scale
                    if hasattr(p, 'x'):
                        logs[f'in/{name}'] = p.x
                    if hasattr(p, 'out'):
                        logs[f'out/{name}'] = p.out
                    if hasattr(p, 'ppl'):
                        logs[f'ppl/{name}'] = p.ppl
                    if hasattr(p, 'agg_ppl'):
                        logs[f'agg_ppl/{name}'] = p.agg_ppl
                for name, p in model.named_modules():
                    if hasattr(p, 'natural_norm'):
                        logs[f'natural_norm/{name}'] = p.natural_norm
                wandb.log(logs)
            pb.set_description(f"Epoch {ep}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")
        if torch.isnan(torch.tensor(train_loss)) and ep >= 5:
            break

    if args.save:
        torch.save(
            model.state_dict(),
            path + "/final_checkpoint.pt",
        )


if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()

    args.num_classes = data_stats.CLASS_DICT[args.dataset]

    if args.n_train is None:
        args.n_train = data_stats.SAMPLE_DICT[args.dataset]

    if args.crop_resolution is None:
        args.crop_resolution = args.resolution

    main(args)
