# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
import os, sys
import pdb
import torch
import torch.nn as nn
import numpy as np
import tqdm
from collections import defaultdict
import gc
from bisect import bisect_right

from lab4d.engine.trainer import Trainer
from lab4d.dataloader import data_utils
from lab4d.utils.torch_utils import get_nested_attr, set_nested_attr

from lab4d.diffgs.gs_model import GSplatModel
from lab4d.diffgs import config


class GSplatTrainer(Trainer):
    def __init__(self, opts):
        """Train and evaluate a Lab4D model.

        Args:
            opts (Dict): Command-line args from absl (defined in lab4d/config.py)
        """
        # if in incremental mode and num-rounds = 0, reset number of rounds
        if opts["inc_warmup_ratio"] > 0 and opts["num_rounds"] == 0:
            eval_dict = self.construct_dataset_opts(opts, is_eval=True)
            evalloader = data_utils.eval_loader(eval_dict)
            warmup_rounds = int(opts["first_fr_steps"] / opts["iters_per_round"])
            inc_rounds = len(evalloader) + warmup_rounds
            opts["num_rounds"] = int(inc_rounds / opts["inc_warmup_ratio"])
            print(f"Num warmup rounds = {warmup_rounds}")
            print(f"Num incremental rounds = {inc_rounds}")
            print(f"Num total rounds = {opts['num_rounds']}")
        super().__init__(opts)

    def define_model(self):
        self.device = torch.device("cuda")
        self.model = GSplatModel(self.opts, self.data_info)
        self.model = self.model.to(self.device)

        self.init_model()

        # cache queue of length 2
        self.model_cache = [None, None]
        self.optimizer_cache = [None, None]
        self.scheduler_cache = [None, None]

        self.grad_queue = {}
        self.param_clip_startwith = {
            # "gaussians._xyz": 5,
            # "gaussians._color_dc": 5,
            # "gaussians._color_rest": 5,
            # "gaussians._scaling": 5,
            # "gaussians._rotation": 5,
            # "gaussians._opacity": 5,
            # "gaussians._trajectory": 5,
            # "gaussians.gs_camera_mlp": 5,
        }

    def load_checkpoint_train(self):
        if self.opts["load_path"] != "":
            # training time
            checkpoint = self.load_checkpoint(
                self.opts["load_path"], self.model, optimizer=self.optimizer
            )
            if not self.opts["reset_steps"]:
                self.current_steps = checkpoint["current_steps"]
                self.current_round = checkpoint["current_round"]
                self.first_round = self.current_round
                self.first_step = self.current_steps

    def optimizer_init(self, is_resumed=False, use_warmup_param=False):
        """Set the learning rate for all trainable parameters and initialize
        the optimizer and scheduler.

        Args:
            is_resumed (bool): True if resuming from checkpoint
        """
        opts = self.opts
        param_lr_startwith, param_lr_with = self.get_lr_dict(
            use_warmup_param=use_warmup_param
        )
        self.params_ref_list, params_list, lr_list = self.get_optimizable_param_list(
            param_lr_startwith, param_lr_with
        )
        self.optimizer = torch.optim.AdamW(
            params_list,
            lr=opts["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        if opts["inc_warmup_ratio"] > 0:
            div_factor = 1.0
            final_div_factor = 25.0
            pct_start = opts["inc_warmup_ratio"]
        else:
            div_factor = 25.0
            final_div_factor = 1.0
            pct_start = min(1 - 1e-5, 0.02)  # use 2% to warm up

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            lr_list,
            int(self.total_steps),
            pct_start=pct_start,
            cycle_momentum=False,
            anneal_strategy="linear",
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

        # # cycle lr
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     self.optimizer,
        #     [i * 0.01 for i in lr_list],
        #     lr_list,
        #     step_size_up=10,
        #     step_size_down=1990,
        #     mode="triangular",
        #     gamma=1.0,
        #     scale_mode="cycle",
        #     cycle_momentum=False,
        # )

    def get_lr_dict(self, use_warmup_param=False):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        opts = self.opts
        lr_base = opts["learning_rate"]

        if use_warmup_param:
            param_lr_startwith = {
                "bg_color": lr_base * 5,
            }
            param_lr_with = {
                "._color_dc": lr_base,
                "._color_rest": lr_base * 0.05,
                "._trajectory": lr_base,
                ".gs_camera_mlp": lr_base * 2,
            }
        else:
            if opts["extrinsics_type"] == "image":
                camera_lr = lr_base * 0.1
                xyz_lr = lr_base * 0.2
            elif opts["extrinsics_type"] == "mlp":
                camera_lr = lr_base
                xyz_lr = lr_base
            else:
                camera_lr = lr_base * 2
                xyz_lr = lr_base
            param_lr_startwith = {
                "bg_color": lr_base * 5,
            }
            param_lr_with = {
                "._xyz": xyz_lr,
                "._color_dc": lr_base,
                "._color_rest": lr_base * 0.05,
                "._scaling": lr_base * 0.5,
                "._rotation": lr_base * 0.5,
                "._opacity": lr_base * 5,
                "._feature": lr_base,
                "._logsigma": lr_base,
                "._trajectory": lr_base * 0.5,
                ".gs_camera_mlp": camera_lr * 0.1,
                ".lab4d_model": lr_base * 0.1,
                ".shadow_field": lr_base * 0.1,
            }

        return param_lr_startwith, param_lr_with

    def train(self):
        super().train()

    def save_checkpoint(self, round_count):
        """Save model checkpoint to disk

        Args:
            round_count (int): Current round index
        """
        opts = self.opts

        if round_count % opts["save_freq"] == 0 or round_count == opts["num_rounds"]:
            # print(f"saving round {round_count}")
            param_path = f"{self.save_dir}/ckpt_{round_count:04d}.pth"

            checkpoint = {
                "current_steps": self.current_steps,
                "current_round": self.current_round,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            torch.save(checkpoint, param_path)
            # copy to latest
            latest_path = f"{self.save_dir}/ckpt_latest.pth"
            os.system(f"cp {param_path} {latest_path}")

    def check_grad(self):
        return {}

    @staticmethod
    def construct_test_model(opts, model_class=GSplatModel, return_refs=True, force_reload=True):
        return Trainer.construct_test_model(opts, model_class=model_class, return_refs=return_refs, force_reload=force_reload)

    def train_one_round(self):
        """Train a single round (going over mini-batches)"""
        opts = self.opts
        gc.collect()  # need to be used together with empty_cache()
        torch.cuda.empty_cache()
        self.model.train()
        self.optimizer.zero_grad()

        # set max loader length for incremental opt
        if opts["inc_warmup_ratio"] > 0:
            self.set_warmup_hparams()
        for i, batch in enumerate(self.trainloader):
            if i == opts["iters_per_round"] * opts["grad_accum"]:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}

            progress = (self.current_steps - self.first_step) / self.total_steps
            sub_progress = i / (opts["iters_per_round"] * opts["grad_accum"])
            self.model.set_progress(self.current_steps, progress, sub_progress)

            loss_dict = self.model(batch)
            total_loss = torch.sum(torch.stack(list(loss_dict.values())))
            total_loss.mean().backward()
            # self.print_sum_params()

            grad_dict = self.check_grad()
            if (i + 1) % opts["grad_accum"] == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # update scalar dict
            # move all to loss
            new_loss_dict = {}
            for k, v in loss_dict.items():
                new_loss_dict[f"loss/{k}"] = v
            del loss_dict
            loss_dict = new_loss_dict
            loss_dict["loss/total"] = total_loss
            loss_dict.update(self.model.get_field_params())
            loss_dict.update(grad_dict)
            self.add_scalar(self.log, loss_dict, self.current_steps)
            self.current_steps += 1

    def set_warmup_hparams(self):
        """Set the loader range for incremental optimization"""
        inc_warmup_ratio = self.opts["inc_warmup_ratio"]
        warmup_rounds = inc_warmup_ratio * self.opts["num_rounds"]

        # config optimizer
        if self.current_round < warmup_rounds:
            self.optimizer_init(use_warmup_param=True)
        elif self.current_round == warmup_rounds:
            self.optimizer_init(use_warmup_param=False)
        self.scheduler.last_epoch = self.current_steps  # specific to onecyclelr

        # config dataloader
        first_fr_steps = self.opts["first_fr_steps"]  # 1st fr warmup steps
        first_fr_ratio = first_fr_steps / (inc_warmup_ratio * self.total_steps)
        completion_ratio = self.current_round / (warmup_rounds - 1)
        completion_ratio = (completion_ratio - first_fr_ratio) / (1 - first_fr_ratio)
        for dataset in self.trainloader.dataset.datasets:
            # per pair opt
            if self.current_round < int(warmup_rounds * first_fr_ratio):
                min_frameid = 0
                max_frameid = 1
            elif self.current_round < warmup_rounds:
                min_frameid = int((len(dataset) - 1) * completion_ratio)
                max_frameid = min_frameid + 1
            else:
                min_frameid = 0
                max_frameid = len(dataset)

            dataset.set_loader_range(min_frameid=min_frameid, max_frameid=max_frameid)
            print(f"setting loader range to {min_frameid}-{max_frameid}")

        # set parameters for incremental opt
        if (
            self.current_round >= int(warmup_rounds * first_fr_ratio)
            and self.current_round < warmup_rounds
        ):
            self.model.gaussians.set_future_time_params(min_frameid)

    def update_aux_vars(self):
        self.model.update_geometry_aux()
        self.model.export_geometry_aux(f"{self.save_dir}/{self.current_round:03d}-all")

        # add some noise to improve convergence
        if self.current_round !=0 and self.current_round % 10 == 0:
            self.model.gaussians.reset_gaussian_scale()
        if self.current_round > 5:
            self.model.gaussians.randomize_gaussian_center()

    def prune_parameters(self, valid_mask, clone_mask):
        """
        Remove the optimizer state of the pruned parameters.
        Set the parameters to the remaining ones.
        """
        # first clone, then prune
        dev = self.device
        clone_mask = torch.logical_and(valid_mask, clone_mask)
        valid_mask = torch.cat(
            (valid_mask, torch.ones(clone_mask.sum(), device=dev).bool())
        )

        for param_dict in self.params_ref_list:
            ((name, _),) = param_dict.items()
            if not "._" in name: # pts related params
                continue
            param = get_nested_attr(self.model, name)
            stored_state = self.optimizer.state.get(param, None)
            if stored_state is not None:
                exp_avg = stored_state["exp_avg"]
                exp_avg_sq = stored_state["exp_avg_sq"]

                exp_avg = torch.cat((exp_avg, exp_avg[clone_mask]))[valid_mask]
                exp_avg_sq = torch.cat((exp_avg_sq, exp_avg_sq[clone_mask]))[valid_mask]

                stored_state["exp_avg"] = exp_avg
                stored_state["exp_avg_sq"] = exp_avg_sq

                del self.optimizer.state[param]
                self.optimizer.state[param] = stored_state

            # set param
            param = torch.cat((param, param[clone_mask]))[valid_mask]
            param = nn.Parameter(param.requires_grad_(True))
            set_nested_attr(self.model, name, param)

    def print_sum_params(self):
        """Print the sum of parameters"""
        sum = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                sum += p.abs().sum()
        print(f"{sum:.16f}")


    def model_eval(self):
        """Evaluate the current model"""
        ref_dict, rendered = super().model_eval()
