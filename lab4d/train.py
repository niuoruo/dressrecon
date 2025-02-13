# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
import os
import sys

import torch
import torch.backends.cudnn as cudnn
from absl import app

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.config import get_config, save_config

cudnn.benchmark = True


def train(Trainer):
    opts = get_config()
    save_config()

    # torch.manual_seed(0)
    # torch.cuda.manual_seed(1)
    # torch.manual_seed(0)

    trainer = Trainer(opts)
    trainer.train()


def main(_):
    from lab4d.engine.trainer import Trainer

    train(Trainer)


if __name__ == "__main__":
    app.run(main)
