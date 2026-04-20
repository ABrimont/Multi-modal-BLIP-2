"""
 Copyright (c) 2022, anonymous.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
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
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
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

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.evaluate(skip_reload=True)
    
    # collect_and_save_attentions_QF(model)


# --- À la fin de evaluate.py ---

def collect_and_save_attentions_QF(model, output_path="attentions_moyennes.pt"):
    final_data = {}
    
    for i, layer in enumerate(model.bert.encoder.layer):
        if hasattr(layer, 'attention_store') and len(layer.attention_store) > 0:
            # On combine tous les batchs [Total_Samples, 48, 48]
            layer_attn = torch.cat(layer.attention_store, dim=0)
            # On calcule la moyenne pour cette couche
            final_data[f"layer_{i}"] = layer_attn.mean(dim=0)
            
    # Sauvegarde sur le disque
    if len(final_data) > 0:
        torch.save(final_data, output_path)
        print(f"✅ Attentions sauvegardées dans {output_path}")


if __name__ == "__main__":
    main()
