import argparse
from .optimizer import OPTIMIZERS_DICT, SCHEDULERS


def get_training_parser():
    parser = argparse.ArgumentParser(description="Scaling MLPs")

    # Data
    parser.add_argument("--data_path", default="./beton", type=str, help="Path to data directory")
    parser.add_argument("--dataset", default="imagenet21", type=str, help="Dataset")
    parser.add_argument("--resolution", default=32, type=int, help="Image Resolution")
    parser.add_argument("--crop_resolution", default=None, type=int, help="Crop Resolution")
    parser.add_argument("--n_train", default=None, type=int, help="Number of samples. None for all")
    parser.add_argument("--ar_modeling", default=False, action=argparse.BooleanOptionalAction, help="Whether to use AR modeling")
    parser.add_argument("--max_freq", default=1, type=int, help="Maximum frequency for the sine waves")
    parser.add_argument("--min_freq", default=0, type=int, help="Minimum frequency for the sine waves")
    parser.add_argument("--input_dim", default=10, type=int, help="Input dimension for the synthetic dataset")



    # Model
    parser.add_argument("--use_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle_pixels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", default="BottleneckMLP", type=str, help="Type of model")
    parser.add_argument("--num_ffn_experts", type=int, default=1)
    parser.add_argument("--width", type=int, default="1024")
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--depth", type=int, default="5")
    parser.add_argument("--heads", type=int, default="8")
    parser.add_argument("--dim_head", type=int, default=None)
    parser.add_argument("--ffn_expansion", type=int, default="4")
    parser.add_argument("--patch_size", type=int, default="8")
    parser.add_argument("--in_channels", type=int, default="3")
    parser.add_argument("--emb_mult", type=float, default=1)
    parser.add_argument("--attn_mult", type=float, default=1)
    parser.add_argument("--output_mult", type=float, default=1)
    parser.add_argument("--fact", type=str, default="tree")
    parser.add_argument("--int_pow", type=float, default=0.0)
    parser.add_argument("--layer_norm", default=True, type=bool)
    parser.add_argument("--expr", type=str, default="")
    parser.add_argument("--init_type", type=str, default="bmm0")

    # CoLA
    parser.add_argument("--cores_n", type=int, default=2)
    parser.add_argument("--use_wrong_mult", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--init_method", type=str, default='µP')

    parser.add_argument("--init_mult_1", type=float, default=1)
    parser.add_argument("--init_mult_2", type=float, default=1)
    parser.add_argument("--lr_mult_1", type=float, default=1)
    parser.add_argument("--lr_mult_2", type=float, default=1)

    parser.add_argument("--input_lr_mult", type=float, default=1)
    parser.add_argument("--init_mult", type=float, default=1)
    parser.add_argument("--struct", type=str, default="none")
    parser.add_argument("--layers", type=str, default="all_but_last")
    parser.add_argument("--rank_frac", type=float, default=0)
    parser.add_argument("--tt_rank", type=int, default=1)
    parser.add_argument("--tt_cores", type=int, default=2)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--aux_loss_weight", type=float, default=0.01)
    parser.add_argument("--spec_penalty_weight", type=float, default=0.)

    # Training
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--optimizer", default="adamw", type=str, help="Choice of optimizer", choices=OPTIMIZERS_DICT.keys())
    parser.add_argument("--batch_size", default=4096, type=int, help="Batch size")
    parser.add_argument("--accum_steps", default=1, type=int, help="Number of accumulation steps")
    parser.add_argument("--lr", default=0.00005, type=float, help="Learning rate")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="Warmup epochs")
    parser.add_argument("--scheduler", type=str, default="none", choices=SCHEDULERS, help="Scheduler")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
    parser.add_argument("--epochs", default=500, type=int, help="Epochs")
    parser.add_argument("--smooth", default=0.3, type=float, help="Amount of label smoothing")
    parser.add_argument("--clip", default=0., type=float, help="Gradient clipping")
    parser.add_argument("--reload", action=argparse.BooleanOptionalAction, default=False, help="Reinitialize from checkpoint")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Whether to augment data")
    parser.add_argument("--mixup", default=0.8, type=float, help="Strength of mixup")
    parser.add_argument("--maximum_steps", default=int(1e10), type=int, help="Maximum steps in synthetic data training")


    # Logging
    parser.add_argument("--calculate_stats", type=int, default=1, help="Frequence of calculating stats")
    parser.add_argument("--checkpoint_folder", type=str, default="./checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--save_freq", type=int, default=50, help="Save frequency")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save checkpoints")
    parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction, help="Whether to log with wandb")
    parser.add_argument("--wandb_project", default="struct_mlp", type=str, help="Wandb project name")
    parser.add_argument("--wandb_entity", default=None, type=str, help="Wandb entity name")

    return parser


def get_finetune_parser():
    parser = argparse.ArgumentParser(description="Scaling MLPs")
    # Data
    parser.add_argument("--data_path", default="./beton", type=str, help="Path to data directory")
    parser.add_argument("--dataset", default="cifar100", type=str, help="Dataset")
    parser.add_argument("--data_resolution", default=64, type=int, help="Image Resolution")
    parser.add_argument("--n_train", default=None, type=int, help="Number of samples. None for all")
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to augment data",
    )
    parser.add_argument("--mixup", default=0., type=float, help="Strength of mixup")
    parser.add_argument('--crop_scale', nargs='+', type=float, default=[0.08, 1.], help="Scale for crop at test time")
    parser.add_argument('--crop_ratio', nargs='+', type=float, default=[0.08, 1.], help="Ratio for crop at test time")
    parser.add_argument("--drop_rate", default=None, type=float, help="Drop rate for dropout")

    # Training
    parser.add_argument(
        "--optimizer",
        default="sgd",
        type=str,
        help="Choice of optimizer",
        choices=OPTIMIZERS_DICT.keys(),
    )
    parser.add_argument("--batch_size", default=4096, type=int, help="Batch size")
    parser.add_argument("--accum_steps", default=1, type=int, help="Number of accumulation steps")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--scheduler", type=str, default="none", choices=SCHEDULERS, help="Scheduler")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
    parser.add_argument("--epochs", default=500, type=int, help="Epochs")
    parser.add_argument("--smooth", default=0.3, type=float, help="Amount of label smoothing")
    parser.add_argument("--clip", default=1.0, type=float, help="Gradient clipping")

    # Misc
    parser.add_argument(
        "--mode",
        default="linear",
        type=str,
        help="Mode",
        choices=["linear", "finetune"],
    )
    parser.add_argument(
        "--checkpoint_folder",
        default="./checkpoints_finetune",
        type=str,
        help="Folder to store checkpoints",
    )
    parser.add_argument("--checkpoint_path", default='checkpoints_finetune', type=str, help="Checkpoint", required=True)
    parser.add_argument(
        "--body_learning_rate_multiplier",
        default=0.1,
        type=float,
        help="Percentage of learning rate for the body",
    )
    parser.add_argument(
        "--calculate_stats",
        type=int,
        default=1,
        help="Frequency of calculating stats",
    )
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save checkpoints",
    )

    return parser
