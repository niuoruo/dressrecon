# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
import os
import sys
import pdb
from absl import app

from lab4d.train import train
from lab4d.diffgs.trainer import GSplatTrainer


def main(_):
    train(GSplatTrainer)


if __name__ == "__main__":
    app.run(main)
