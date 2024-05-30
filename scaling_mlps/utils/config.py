import os
import json


def config_to_name(args):
    cola = args.struct
    if args.struct == "low_rank":
        cola = f"{cola}_rank{args.rank_frac}"
    cola = os.path.join(cola, args.layers)
    return os.path.join(args.dataset, f"{args.model}_d{args.depth}_w{args.width}", cola, f"{args.optimizer}_lr{args.lr}")


def model_from_config(path):
    """Return model class from checkpoint path."""
    path = os.path.dirname(path)
    with open(path + '/config.txt', 'r') as f:
        config = json.load(f)
    model = config["model"]
    architecture = config["architecture"]
    norm = config["normalization"]
    crop_resolution = int(config["crop_resolution"])

    return model, architecture, crop_resolution, norm
