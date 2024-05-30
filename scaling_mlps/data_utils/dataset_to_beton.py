import argparse
import os

import torchvision
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter
from tqdm import tqdm
import numpy as np
import torch

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_dataset(dataset_name, mode, data_path):
    if data_path is not None:
        return torchvision.datasets.ImageFolder(root=data_path, transform=None)

    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(root="/tmp", train=mode == "train", download=True)
    elif dataset_name == "cifar100":
        return torchvision.datasets.CIFAR100(root="/tmp", train=mode == "train", download=True)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported. Please add it here.")


def create_beton(args):
    dataset = get_dataset(args.dataset_name, args.mode, args.data_path)

    write_path = os.path.join(args.write_path, args.dataset_name, args.mode, f"{args.mode}_{args.res}.beton")

    os.makedirs(os.path.dirname(write_path), exist_ok=True)

    writer = DatasetWriter(
        write_path,
        {
            "image": RGBImageField(write_mode="smart", max_resolution=args.res),
            "label": IntField(),
        },
        num_workers=args.num_workers,
    )

    writer.from_indexed_dataset(dataset, chunksize=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to .beton format")
    parser.add_argument("--dataset_name", type=str, default=None, help="dataset name")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="path to dataset if data is given in a hierarchical subfolder structure.",
    )
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--res", type=int, default=32, help="resolution of images")
    parser.add_argument(
        "--write_path",
        type=str,
        default="./beton/",
        help="path to write .beton file to",
    )
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers to use")
    args = parser.parse_args()

    assert (args.dataset_name or args.data_path), "Either dataset_name or data_path must be specified."

    create_beton(args)
