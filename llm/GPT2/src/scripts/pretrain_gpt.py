"""
The script will:
1. Load data from the specified dataset
2. Create train/validation splits
3. Initialize the GPT model
4. Train the model with mixed precision
5. Save checkpoints and log to wandb

"""

import os
import math
import numpy as np
import random
import logging
import argparse
from typing import Optional, Callable, List, Tuple, Dict, Any

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from torch.amp import autocast, GradScaler

# Data loading imports
from torch.utils.data import Dataset, DataLoader
import json
import glob
import gzip
import bz2
import datetime

# Arrow dataset support
from datasets import load_from_disk

# Tokenization imports
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Progress and timing
from tqdm.auto import tqdm, trange
import time
import wandb

# Import our GPT implementation
import gpt
from sklearn.model_selection import train_test_split

# Set CuPy/CUDA to allow TF32 computations
# This can provide a speedup on compatible GPUs (RTX 4000 series, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT model')

    # Data arguments
    parser.add_argument('--data_path', type=str,
                       default='./Data/fineweb-edu-sample-1M.jsonl.gz',
                       help='Path to the training data (JSONL.gz file or Arrow dataset directory)')
    parser.add_argument('--data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of training data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--max_docs', type=int, default=None,
                       help='Maximum number of documents to load (for testing, only applies to raw text)')

    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=None,
                       help='Vocabulary size (auto-detected if not specified)')
    parser.add_argument('--context_length', type=int, default=1024,
                       help='Context length')
    parser.add_argument('--emb_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=2,
                       help='Maximum number of epochs')
    parser.add_argument('--target_tokens', type=int, default=1_200_000_000,
                       help='Target number of tokens to train on')

    # Validation arguments
    parser.add_argument('--eval_data_path', type=str, default='./Data/fineweb-edu-eval-100K.jsonl.gz',
                       help='Path to validation data')
    parser.add_argument('--eval_data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of validation data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--eval_max_docs', type=int, default=None,
                       help='Maximum number of documents to load for validation (only for raw text)')
    parser.add_argument('--eval_max_docs_step', type=int, default=None,
                       help='Maximum number of validation documents to use during step evaluation (None = use all)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                       help='Validation batch size')


    # Logging and saving
    parser.add_argument('--output_dir', type=str,
                       default='./models/pretrained-models/',
                       help='Output directory for saving models')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='Save model every N steps')
    parser.add_argument('--eval_every', type=int, default=1000,
                       help='Evaluate model every N steps')
    parser.add_argument('--wandb_project', type=str, default='gpt-pretraining',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str,
                       default=f"gpt-pretraining-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                       help='Wandb run name')
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    """Determine the best available device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    else:
        return device_arg

def get_amp_dtype(device):
    '''Get the appropriate AMP dtype for mixed precision training on the device.'''

    if device.startswith('cuda'):
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == 'mps':
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32  # or disable autocast on CPU
    return amp_dtype

def load_data(data_path, max_docs=None, data_format='jsonl'):
    """
    Load data from JSONL file or Arrow dataset.

    Args:
        data_path: Path to the data file or Arrow dataset directory
        max_docs: Maximum number of documents to load (only for raw text)
        data_format: Format of the data ('jsonl' or 'arrow')
    Returns:
        List of text documents (for raw text) or None (for Arrow datasets)
    """
    if data_format == 'arrow':
        print(f"Using Arrow dataset from {data_path}")
        # For Arrow datasets, we don't need to load the data here
        # The GPTArrowDataset in gpt.py will handle loading
        return None
    else:
        print(f"Loading data from {data_path}")

        ofunc = gzip.open if data_path.endswith('gz') else open
        docs = []

        with ofunc(data_path, 'rt') as f:
            for i, line in enumerate(tqdm(f, desc="Reading data from file")):
                if max_docs and i >= max_docs:
                    break
                docs.append(json.loads(line)['text'])

        print(f"Loaded {len(docs)} documents")
        return docs
