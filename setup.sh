# CIFAR-10/100 and ImageNet experiments
# We need to install FFCV, which is pretty painful and it's going to take a while
conda env create -f conda_env.yml --force # env name will be "struct"
conda activate struct
pip install --upgrade pip
pip uninstall torch -y
pip uninstall torchvision -y
pip install torch==1.12.0 torchvision==0.13.0 torchdistx --index-url https://download.pytorch.org/whl/cu116
conda install -c pytorch -c conda-forge torchdistx cudatoolkit=11.6 --force
cd ~; git clone https://github.com/wilson-labs/cola.git; cd -
sed -i '/torch\.func/s/^/#/' ~/cola/cola/backends/torch_fns.py # comment out lines containing torch.func which is not available in torch 1.12
pip install $HOME/cola/.[dev]
pip install datasets tiktoken torchmetrics
conda deactivate

# GPT experiments
conda create -n gpt python=3.10 --force
conda activate gpt
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy transformers datasets tiktoken wandb tqdm torchmetrics
cd ~; git clone https://github.com/wilson-labs/cola.git; cd -
pip install $HOME/cola/.[dev]
pip install -U fvcore torchinfo einops
conda deactivate