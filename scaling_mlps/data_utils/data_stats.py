import numpy as np

# Define all the relevant stats for the datasets to look up

# Number of samples
SAMPLE_DICT = {
    "imagenet21": 11801680,
    "imagenet": 1281167,
    "imagenet_real": 1281167,
    "tinyimagenet": 100000,
    "cifar10": 50000,
    "cifar5m": 5982688,
    "cifar100": 50000,
    "stl10": 5000,
    "synthetic": 50000000,
    "sine_waves": int(1e9)
}

# Number of classes
CLASS_DICT = {
    "imagenet21": 11230,
    "imagenet": 1000,
    'imagenet_real': 1000,
    "tinyimagenet": 200,
    "cifar10": 10,
    "cifar5m": 10,
    "cifar100": 100,
    "stl10": 10,
    "synthetic": 1,
    "sine_waves": 1
}

# Image resolutions
DEFAULT_RES_DICT = {
    "imagenet21": 64,
    "imagenet": 64,
    "imagenet_real": 64,
    "tinyimagenet": 64,
    "cifar10": 32,
    "cifar5m": 32,
    "cifar100": 32,
    "stl10": 64,
}

# Parent directory name
DATA_DICT = {
    "imagenet21": "imagenet21",
    "imagenet": "imagenet",
    "tinyimagenet": "tiny-imagenet-200",
    "imagenet_real": "imagenet",
    "cifar10": "cifar10",
    "cifar5m": "cifar5m",
    "cifar100": "cifar100",
    "stl10": "stl10",
}

MODE_DICT = {
    "imagenet21": "test",
    "imagenet": "val",
    'imagenet_real': "val",
    "tinyimagenet": "val",
    "cifar10": "val",
    "cifar100": "val",
    "stl10": "val",
}

# Standardization statistics
MEAN_DICT = {
    "imagenet21": np.array([0.485, 0.456, 0.406]) * 255,
    "imagenet": np.array([0.485, 0.456, 0.406]) * 255,
    "imagenet_real": np.array([0.485, 0.456, 0.406]) * 255,
    "tinyimagenet": np.array([0.485, 0.456, 0.406]) * 255,
    "cifar10": np.array([0.49139968, 0.48215827, 0.44653124]) * 255,
    "cifar5m": np.array([0.49139968, 0.48215827, 0.44653124]) * 255,
    "cifar100": np.array([0.49139968, 0.48215827, 0.44653124]) * 255,
    "stl10": np.array([0.4914, 0.4822, 0.4465]) * 255,
}

STD_DICT = {
    "imagenet21": np.array([0.229, 0.224, 0.225]) * 255,
    "imagenet": np.array([0.229, 0.224, 0.225]) * 255,
    "imagenet_real": np.array([0.229, 0.224, 0.225]) * 255,
    "tinyimagenet": np.array([0.229, 0.224, 0.225]) * 255,
    "cifar10": np.array([0.24703233, 0.24348505, 0.26158768]) * 255,
    "cifar5m": np.array([0.24703233, 0.24348505, 0.26158768]) * 255,
    "cifar100": np.array([0.24703233, 0.24348505, 0.26158768]) * 255,
    "stl10": np.array([0.2471, 0.2435, 0.2616]) * 255,
}

# Whether dataset can be cached in memory
OS_CACHED_DICT = {
    "imagenet21": False,
    "imagenet": False,
    "imagenet_real": False,
    "tinyimagenet": True,
    "cifar10": True,
    "cifar5m": True,
    "cifar100": True,
    "stl10": True,
    "pets": True,
    "coyo": False,
}
