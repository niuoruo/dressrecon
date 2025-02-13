# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
# python scripts/render.py --seqname --flagfile=logdir/cat-0t10-fg-bob-d0-long/opts.log --load_suffix latest

import os, sys
from absl import app

from lab4d.render import render, get_config, construct_batch_from_opts
from lab4d.diffgs.trainer import GSplatTrainer as Trainer


def main(_):
    opts = get_config()
    render(opts, construct_batch_func=construct_batch_from_opts, Trainer=Trainer)


if __name__ == "__main__":
    app.run(main)
