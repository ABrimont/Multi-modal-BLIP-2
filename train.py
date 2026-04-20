"""
 Copyright (c) 2022, anonymous.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os

# ⚡ Forcer les chemins HuggingFace AVANT tout import transformers
# cache_dir = "/home/abrimont/partage/mllm-video-captioner/.cache/huggingface"
# os.environ["HF_HOME"] = cache_dir
# os.environ["TRANSFORMERS_CACHE"] = cache_dir
# os.environ["XDG_CACHE_HOME"] = "/home/abrimont/partage/mllm-video-captioner/.cache"

# print(">>> HF_HOME forcé à:", os.environ["HF_HOME"])


import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()

if __name__ == "__main__":
    import os
    import torch.multiprocessing as mp

    # # --- Debug + stabilité NCCL ---
    # os.environ.setdefault("NCCL_DEBUG", "INFO")
    # os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    # os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # os.environ.setdefault("NCCL_P2P_DISABLE", "1")     # utile sur RTX 4090
    # os.environ.setdefault("NCCL_IB_DISABLE", "1")
    # os.environ.setdefault("NCCL_SHM_DISABLE", "1")

    # # --- CUDA Debug (OOM explicites) ---
    # os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
    # os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    # --- Tokenizers / Threads ---
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"

    # --- Multiprocessing ---
    # mp.set_start_method("spawn", force=True)
    # mp.set_sharing_strategy("file_descriptor")

    # --- Launch main training ---
    main()
