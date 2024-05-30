import os
from urllib.request import urlretrieve
import progressbar

default_checkpoints = {
    'B-12_Wi-1024_res_64': 'B-12_Wi-1024_res_64_imagenet21_epochs_800',
    'B-12_Wi-512_res_64': 'B-12_Wi-512_res_64_imagenet21_epochs_600',
    'B-6_Wi-1024_res_64': 'B-6_Wi-1024_res_64_imagenet21_epochs_800',
    'B-6_Wi-512_res_64': 'B-6_Wi-512_res_64_imagenet21_epochs_800',
    'B-12_Wi-1024_res_64_cifar10': 'B-12_Wi-1024_res_64_cifar10_epochs_20',
    'B-12_Wi-1024_res_64_cifar100': 'B-12_Wi-1024_res_64_cifar100_epochs_40',
    'B-12_Wi-1024_res_64_imagenet': 'B-12_Wi-1024_res_64_imagenet_epochs_50'
}

weight_urls = {
    'B-12_Wi-1024_res_64_imagenet21_epochs_800':
    'https://drive.google.com/uc?id=1rcV8RXij_kW9X2zSLNyNOTO_bKUjE0cJ&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_imagenet21_epochs_600':
    'https://drive.google.com/uc?id=1sL9j_4FFeBTWTzuRFbHLfwrmPVLXhUtW&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_imagenet21_epochs_800':
    'https://drive.google.com/uc?id=1cmO3QSz_hfHtyzkUOZnPmXPxIE2YzEbf&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_imagenet21_epochs_800':
    'https://drive.google.com/uc?id=1QV3a99UT8llfh9zDDuNKWDH_5c6S_YT5&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_cifar10_epochs_20':
    'https://drive.google.com/uc?id=1GyxuewoOzMRhzEOyUrIBLQzquc-QEYNV&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_cifar100_epochs_40':
    'https://drive.google.com/uc?id=1LNqC58cSwtuDr-C4bk1O3GA_vAWls-UH&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_imagenet_epochs_50':
    'https://drive.google.com/uc?id=1MVebvnSGL02k_ql1gUCjh4quGqM9RM4F&export=download&confirm=t&uuid'
}

config_urls = {
    'B-12_Wi-1024_res_64_imagenet21_epochs_800':
    'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_imagenet21_epochs_600':
    'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_imagenet21_epochs_800':
    'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_imagenet21_epochs_800':
    'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_cifar10_epochs_20':
    'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_cifar100_epochs_40':
    'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_imagenet_epochs_50':
    'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid'
}


def download(name, checkpoint_path):
    weight_url = weight_urls[name]
    config_url = config_urls[name]

    weight_path = checkpoint_path + name + '_weights'
    config_path = checkpoint_path + name + '_config'
    weight_exists = os.path.isfile(weight_path)
    config_exists = os.path.isfile(config_path)

    if not weight_exists:
        print('Downloading weights...')
        urlretrieve(weight_url, weight_path, show_progress)
    else:
        print('Weights already downloaded')
    if not config_exists:
        urlretrieve(config_url, config_path)

    return weight_path, config_path


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None
