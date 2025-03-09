import numpy as np
import argparse
import tqdm
import random
import logging
from tqdm import tqdm
import model_driver
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
from torchvision import datasets, transforms
from torch.optim import AdamW, Adam
from torch.cuda.amp import autocast, GradScaler

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from functools import partial

from PIL import ImageFilter, ImageOps, Image

from ignite.utils import convert_tensor

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from genetic_algorithm_evolution import GeneticAlgorithmEvolution 
from utils.dataloader import dataload
from src.swin_vit import SwinTransformer
from utils.scheduler import build_scheduler  
from utils.dataloader import datainfo
from utils.optimizer import get_adam_optimizer
from utils.utils import clip_gradients
from utils.utils import save_checkpoint
from utils.cutmix import CutMix
from model_validator import ModelValidator
import warnings
from hybrid_cmaes_with_de import HybridCMAESWithDE
warnings.filterwarnings("ignore")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    parser = argparse.ArgumentParser('SWIN ViT for CIFAR-10', add_help=False)
    parser.add_argument('--dir', type=str, default='./data',
                    help='Data directory')
    parser.add_argument('--num_classes', type=int, default=10, choices=[10, 100, 1000],
                    help='Dataset name')

    # Model parameters
    parser.add_argument('--patch_size', default=2, type=int, help="""Size in pixels of input square patches - default 4 (for 4x4 patches) """)
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of the SSL MLP head output. For complex and large datasets large values (like 65k) work well.""")

    parser.add_argument('--norm_last_layer', default=False, type=bool,
        help="""Whether or not to weight normalize the last layer of the MLP head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool,
        help="Whether to use batch normalizations in projection head (Default: False)")

    parser.add_argument('--image_size', default=32, type=int, help=""" Size of input image. """)
    parser.add_argument('--in_channels',default=3, type=int, help=""" input image channels. """)
    parser.add_argument('--embed_dim',default=192, type=int, help=""" dimensions of vit """)
    parser.add_argument('--num_layers',default=9, type=int, help=""" No. of layers of ViT """)
    parser.add_argument('--num_heads',default=12, type=int, help=""" No. of heads in attention layer
                                                                                    in ViT """)
    parser.add_argument('--vit_mlp_ratio',default=2, type=int, help=""" MLP hidden dim """)
    parser.add_argument('--qkv_bias',default=True, type=bool, help=""" Bias in Q K and V values """)
    parser.add_argument('--drop_rate',default=0., type=float, help=""" dropout """)

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=1e-1, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--batch_size', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. Recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing for optimizer')
    parser.add_argument('--gamma', type=float, default=1.0,
                    help='Gamma value for Cosine LR schedule')

    # Misc
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100'], help='Please specify path to the training data.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--mlp_head_in", default=192, type=int, help="input dimension going inside MLP projection head")
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="directory to save checkpoints")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("\n--- GPU Information ---\n")

    if torch.cuda.is_available():
        print(f"Model is using device: {device}")
        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 2} MB")
    else:
        print("Model is using CPU")

    print("\n--- Downloading Data ---\n")

    data_info = datainfo(args)
    train_dataset, val_dataset = dataload(args, data_info)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)
    checkpoint = torch.load("experiments/checkpoints/model_199.pt", map_location=torch.device('cpu'))

    # Step 2: Load model architecture (if necessary)
    # If the checkpoint contains only the state_dict:
    # Define your model architecture
    # model = MyModel()
    # model.load_state_dict(checkpoint)

    # If the checkpoint contains the entire model:
    model = checkpoint  # Assuming the saved file has the model itself
    state_dict = model
    # Step 3: Inspect the keys in the state_dict to identify the last layer
    # print("Keys in state_dict:", state_dict.keys())

    # Step 4: Access the last layer's weights
    # Replace 'fc.weight' with the actual key for the last layer
    last_layer_key = list(state_dict.keys())[-2]  # Assuming the last key corresponds to the last layer
    print("Last layer key:", last_layer_key)

    last_layer_weights = state_dict[last_layer_key]
    print("Last layer weights:", last_layer_weights)

    model = SwinTransformer(img_size=args.image_size,
                        num_classes=args.num_classes,
                        window_size=4, 
                        patch_size=args.patch_size, 
                        embed_dim=96, 
                        depths=[2, 6, 4], 
                        num_heads=[3, 6, 12],
                        mlp_ratio=args.vit_mlp_ratio, 
                        qkv_bias=True, 
                        drop_path_rate=args.drop_path_rate).to(device)
    last_layer_weights = state_dict[last_layer_key]
    weights_flattened_length = last_layer_weights.numel()
    print(f"Flattened length of weights: {weights_flattened_length}")
    flattened_weights = last_layer_weights.flatten()  # Flattens the tensor into a 1D tensor
    # Convert to a Python list if required
    flattened_weights_list = flattened_weights.tolist()
    shape = last_layer_weights.shape
    print(f"Shape = {shape}")
    last_layer_min = min(flattened_weights_list)
    last_layer_max = max(flattened_weights_list)
    print(f"Maximum = {last_layer_max}, Minimum = {last_layer_min}")
    # print(flattened_weights_list)
    driver = model_driver.ModelDriver("experiments/checkpoints/model_199.pt", model)
    driver.load_checkpoint()
    driver.remove_last_layer_weights()

    print(driver.model)

    validator = ModelValidator(driver.model, train_loader, val_loader)  # Pass the model and val_loader to the validator

    # # Initialize Genetic Algorithm
    # ga = GeneticAlgorithmEvolution(
    #     min_weight=last_layer_min,
    #     max_weight=last_layer_max,
    #     last_layer_shape=shape,
    #     validation_class=validator,  # Pass the Model Validator instance
    # )

    # # Run the Genetic Algorithm
    # best_weights = ga.run(population_size=100, num_generations=50, stop_threshold=93)

    # Hybrid CMA-ES with DE
    hybrid_optimizer = HybridCMAESWithDE(
    min_weight=-0.2,
    max_weight=0.2,
    last_layer_shape=(3840,),
    validation_class=validator,
    de_rate=0.2
    )

    best_weights = hybrid_optimizer.run(
    population_size=50,
    num_generations=100,
    stop_threshold=95,
    save_interval=10,  # Save every 10 generations
    checkpoint_file="hybrid_cmaes_de_checkpoint.pt"
    )

    # Apply the best weights to the model
    driver.model.head.weight.data = torch.tensor(best_weights).reshape(shape).to(device)
    print("Best weights applied to the model.")


