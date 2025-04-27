# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
import os
import time
from collections import defaultdict
from copy import deepcopy
import gc
import configparser

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm
import trimesh
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True

from lab4d.dataloader import data_utils
from lab4d.dataloader.vidloader import VidDataset
from lab4d.engine.model import dvr_model
from lab4d.engine.train_utils import match_param_name
from lab4d.utils.vis_utils import img2color, make_image_grid


class Trainer:
    def __init__(self, opts):
        """Train and evaluate a Lab4D model.

        Args:
            opts (Dict): Command-line args from absl (defined in lab4d/config.py)
        """
        self.opts = opts

        # 读取数据集并且划分为train_dict eval_dict
        self.define_dataset()
        # 初始化标识符，文件夹等
        self.trainer_init()
        # 初始化模型
        self.define_model()

        self.optimizer_init(is_resumed=opts["load_path"] != "")

        # load model
        self.load_checkpoint_train()

    def trainer_init(self):
        """Initialize logger and other misc things"""
        opts = self.opts

        logname = f"{opts['seqname']}-{opts['logname']}"
        self.save_dir = os.path.join(opts["logroot"], logname)

        os.makedirs("tmp/", exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        # tensorboard
        self.log = SummaryWriter(f"{opts['logroot']}/{logname}", comment=logname)

        self.current_steps = 0  # 0-total_steps
        self.current_round = 0  # 0-num_rounds
        self.first_round = 0  # 0
        self.first_step = 0  # 0

        # Select 9 images evenly spaced from eval dataset
        self.eval_fid = np.linspace(0, len(self.evalloader) - 1, 9).astype(int)

        # torch.manual_seed(8)  # do it again
        # torch.cuda.manual_seed(1)

    def define_dataset(self):
        """Construct training and evaluation dataloaders."""
        opts = self.opts

        # Create uncertainty map for each video
        # Initialize pixels to have uncertainty=1, which is considered high
        self.uncertainty_map = {}
        config = configparser.RawConfigParser()
        config.read(f"database/configs/{opts['seqname']}.config")
        for vidid in range(len(config.sections()) - 1):
            img_path = str(config.get(f"data_{vidid}", "img_path")).strip("/")
            vidname = img_path.split("/")[-1]
            uncertainty_name = f"{vidname}-{opts['data_prefix']}-{opts['train_res']}"
            # 从对应npy中读取图像信息 不止RGB，以及提取的feature, mask, flow...
            rgb_path = f"{img_path}/{opts['data_prefix']}-{opts['train_res']}.npy"

            # 从图像信息中获取图像的大小 N, H, W
            shape = np.load(rgb_path, mmap_mode="r").shape[:-1]  # N, H, W
            uncertainty = np.ones(shape, dtype=np.float32)
            self.uncertainty_map[uncertainty_name] = uncertainty

        # 划分训练集和验证集 训练集会标记为crop表示裁剪 验证机标记为full表示全分辨率
        train_dict = self.construct_dataset_opts(opts)
        self.trainloader = data_utils.train_loader(train_dict, self.uncertainty_map)

        eval_dict = self.construct_dataset_opts(opts, is_eval=True)
        self.evalloader = data_utils.eval_loader(eval_dict, self.uncertainty_map)

        self.data_info, self.data_path_dict = data_utils.get_data_info(self.evalloader)

        # 计算训练轮次
        self.total_steps = opts["num_rounds"] * min(
            opts["iters_per_round"] * opts["grad_accum"], len(self.trainloader)
        )

    def init_model(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics from external priors, if this is the first run"""
        # init mlp
        self.model.mlp_init()

    def define_model(self, model=dvr_model):
        """Define a Lab4D model"""
        opts = self.opts
        data_info = self.data_info

        self.device = torch.device("cuda")
        self.model = model(opts, data_info, self.uncertainty_map)
        self.model = self.model.to(self.device)

        self.init_model()

        # 储存训练信息
        # cache queue of length 2
        self.model_cache = [None, None]
        self.optimizer_cache = [None, None]
        self.scheduler_cache = [None, None]

        self.grad_queue = {}
        
        # 梯度裁剪参数 防止梯度爆炸
        self.param_clip_startwith = {
            "fields.field_params.fg.camera_mlp": 10.0,
            "fields.field_params.fg.warp.articulation": 10.0,
            "fields.field_params.fg.basefield": 10.0,
            "fields.field_params.fg.sdf": 10.0,
            "fields.field_params.bg.camera_mlp": 10.0,
            "fields.field_params.bg.basefield": 10.0,
            "fields.field_params.bg.sdf": 10.0,
        }

    def get_lr_dict(self, pose_correction=False):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        opts = self.opts
        lr_base = opts["learning_rate"]
        lr_explicit = lr_base * 10
        lr_intrinsics = 0.0 if opts["freeze_intrinsics"] else lr_base

        param_lr_startwith = {
            "fields.field_params": lr_base,
            "intrinsics": lr_intrinsics,
        }
        param_lr_with = {
            ".logibeta": lr_explicit,
            ".logsigma": lr_explicit,
            ".logscale": lr_explicit,
            ".log_gauss": lr_explicit,
            ".base_quat": 0.0,
            ".shift": lr_explicit,
            ".orient": lr_explicit,
        }

        if pose_correction:
            del param_lr_with[".logscale"]
            del param_lr_with[".log_gauss"]
            param_lr_with_pose_correction = {
                "fields.field_params.fg.basefield.": 0.0,
                "fields.field_params.fg.sdf.": 0.0,
                "fields.field_params.fg.feature_field": 0.0,
                "fields.field_params.fg.warp.skinning_model": 0.0,
            }
            param_lr_with.update(param_lr_with_pose_correction)

        return param_lr_startwith, param_lr_with

    def optimizer_init(self, is_resumed=False):
        """Set the learning rate for all trainable parameters and initialize
        the optimizer and scheduler.

        Args:
            is_resumed (bool): True if resuming from checkpoint
        """
        opts = self.opts
        param_lr_startwith, param_lr_with = self.get_lr_dict(
            pose_correction=self.opts["pose_correction"]
        )
        self.params_ref_list, params_list, lr_list = self.get_optimizable_param_list(
            param_lr_startwith, param_lr_with
        )

        self.optimizer = torch.optim.AdamW(
            params_list,
            lr=opts["learning_rate"],
            betas=(0.9, 0.99),
            weight_decay=1e-4,
        )

        # # one cycle lr
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     lr_list,
        #     int(self.total_steps),
        #     pct_start=min(1 - 1e-5, 10.0 / opts["num_rounds"]),  # use 10 epochs to warm up
        #     cycle_momentum=False,
        #     anneal_strategy="linear",
        #     div_factor=25.0,
        #     final_div_factor=1.0,
        # )

        # cyclic lr
        assert self.total_steps // 2000 * 2000 == self.total_steps # dividible by 2k
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            [i * 0.01 for i in lr_list],
            lr_list,
            step_size_up=10,
            step_size_down=1990,
            mode="triangular",
            gamma=1.0,
            scale_mode="cycle",
            cycle_momentum=False,
        )

    def get_optimizable_param_list(self, param_lr_startwith, param_lr_with):
        """
        Get the optimizable param list
        Returns:
            params_ref_list (List): List of params
            params_list (List): List of params
            lr_list (List): List of learning rates
        """
        params_ref_list = []
        params_list = []
        lr_list = []

        params_out_str = []
        for name, p in self.model.named_parameters():
            matched_loose, lr_loose = match_param_name(name, param_lr_with, type="with")
            matched_strict, lr_strict = match_param_name(
                name, param_lr_startwith, type="startwith"
            )
            if matched_loose > 0:
                lr = lr_loose  # higher priority
            elif matched_strict > 0:
                lr = lr_strict
            else:
                lr = 0.0  # not found
                # print(name, "not found")
            if lr > 0:
                params_ref_list.append({name: p})
                params_list.append({"params": p})
                lr_list.append(lr)
                params_out_str.append(f"[{self.opts['seqname']}] {name}: {p.shape} {lr}")

        with open(f"{self.save_dir}/params.txt", "w") as f:
            f.write("\n".join(params_out_str))

        return params_ref_list, params_list, lr_list

    def train(self):
        """Training loop"""
        opts = self.opts

        # clear buffers for pytorch1.10+
        try:
            self.model._assign_modules_buffers()
        except:
            pass

        # start training loop
        self.save_checkpoint(round_count=self.current_round)
        for round_idx in tqdm.trange(
            self.current_round, self.current_round + opts["num_rounds"],
            desc=f"Training {self.opts['seqname']}"
        ):
            with torch.autograd.set_detect_anomaly(opts["detect_anomaly"]):
                self.run_one_round()
            self.save_checkpoint(round_count=self.current_round)

    def update_aux_vars(self):
        self.model.update_geometry_aux()
        self.model.export_geometry_aux(f"{self.save_dir}/{self.current_round:03d}")
        if (
            self.current_round > self.opts["num_rounds_cam_init"]
            and self.opts["absorb_base"]
        ):
            self.model.update_camera_aux()

    def run_one_round(self):
        """Evaluation and training for a single round"""
        if self.current_round == self.first_round:
            self.model_eval()

        self.update_aux_vars()

        self.model.train()
        self.train_one_round()
        self.current_round += 1
        self.model_eval()

    def save_checkpoint(self, round_count):
        """Save model checkpoint to disk

        Args:
            round_count (int): Current round index
        """
        opts = self.opts
        # move to the left
        self.model_cache[0] = self.model_cache[1]
        self.optimizer_cache[0] = self.optimizer_cache[1]
        self.scheduler_cache[0] = self.scheduler_cache[1]
        # enqueue
        self.model_cache[1] = deepcopy(self.model.state_dict())
        self.optimizer_cache[1] = deepcopy(self.optimizer.state_dict())
        self.scheduler_cache[1] = deepcopy(self.scheduler.state_dict())

        if round_count % opts["save_freq"] == 0 or round_count == opts["num_rounds"]:
            # print(f"[{opts['seqname']}] saving round {round_count}")
            param_path = f"{self.save_dir}/ckpt_{round_count:04d}.pth"

            checkpoint = {
                "current_steps": self.current_steps,
                "current_round": self.current_round,
                "model": self.model_cache[1],
                "optimizer": self.optimizer_cache[1],
                "scheduler": self.scheduler_cache[1],
            }

            torch.save(checkpoint, param_path)
            # copy to latest
            latest_path = f"{self.save_dir}/ckpt_latest.pth"
            os.system(f"cp {param_path} {latest_path}")

            # Flush uncertainty to disk
            os.makedirs(f"{self.save_dir}/uncertainty", exist_ok=True)
            for k, v in self.uncertainty_map.items():
                np.save(f"{self.save_dir}/uncertainty/{k}.npy", v)

    @staticmethod
    def load_checkpoint(load_path, model, optimizer=None):
        """Load a model from checkpoint

        Args:
            load_path (str): Path to checkpoint
            model (dvr_model): Model to update in place
            optimizer (torch.optim.Optimizer or None): If provided, load
                learning rate from checkpoint
        """
        checkpoint = torch.load(load_path, weights_only=True)
        model_states = checkpoint["model"]
        model.load_state_dict(model_states, strict=False)

        # reset near_far
        if hasattr(model, "fields"):
            model.fields.reset_geometry_aux()

        return checkpoint

    def load_checkpoint_train(self):
        """Load a checkpoint at training time and update the current step count
        and round count
        """
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

    def train_one_round(self):
        """Train a single round (going over mini-batches)"""
        opts = self.opts
        gc.collect()  # need to be used together with empty_cache()
        torch.cuda.empty_cache()
        self.model.train()
        self.optimizer.zero_grad()

        for i, batch in enumerate(self.trainloader):
            if i == opts["iters_per_round"] * opts["grad_accum"]:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}

            progress = (self.current_steps - self.first_step) / self.total_steps
            self.model.set_progress(self.current_steps, progress)

            loss_dict = self.model(batch)
            total_loss = torch.sum(torch.stack(list(loss_dict.values())))
            total_loss.mean().backward()
            # print(total_loss)

            grad_dict = self.check_grad()
            if (i + 1) % opts["grad_accum"] == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # update scalar dict
            loss_dict["loss/total"] = total_loss
            loss_dict.update(self.model.get_field_betas())
            loss_dict.update(grad_dict)
            self.add_scalar(self.log, loss_dict, self.current_steps)

            self.current_steps += 1

    @staticmethod
    def construct_dataset_opts(opts, is_eval=False, dataset_constructor=VidDataset):
        """Extract train/eval dataloader options from command-line args.

        Args:
            opts (Dict): Command-line options
            is_eval (bool): When training a model (`is_eval=False`), duplicate
                the dataset to fix the number of iterations per round
            dataset_constructor (torch.utils.data.Dataset): Dataset class to use
        """
        opts_dict = {}
        opts_dict["logroot"] = opts["logroot"]
        opts_dict["seqname"] = opts["seqname"]
        opts_dict["logname"] = opts["logname"]
        opts_dict["uncertainty_factor"] = opts["uncertainty_factor"]
        opts_dict["load_pair"] = True
        opts_dict["data_prefix"] = f"{opts['data_prefix']}-{opts['train_res']}"
        opts_dict["feature_type"] = opts["feature_type"]
        opts_dict["field_type"] = opts["field_type"]
        opts_dict["eval_res"] = opts["eval_res"]
        opts_dict["dataset_constructor"] = dataset_constructor

        if is_eval:
            opts_dict["multiply"] = False
            opts_dict["pixels_per_image"] = -1
            opts_dict["delta_list"] = []
            opts_dict["uncertainty"] = False
        else:
            # duplicate dataset to fix number of iterations per round
            opts_dict["multiply"] = True
            opts_dict["pixels_per_image"] = opts["pixels_per_image"]
            opts_dict["delta_list"] = [2, 4, 8]
            opts_dict["uncertainty"] = True
            opts_dict["num_workers"] = opts["num_workers"]

            opts_dict["imgs_per_gpu"] = opts["imgs_per_gpu"]
            opts_dict["iters_per_round"] = opts["iters_per_round"]
            opts_dict["grad_accum"] = opts["grad_accum"]
            opts_dict["ngpu"] = opts["ngpu"]
        return opts_dict

    @torch.no_grad()
    def model_eval(self):
        """Evaluate the current model"""
        self.model.eval()
        gc.collect()  # need to be used together with empty_cache()
        torch.cuda.empty_cache()
        ref_dict, batch = self.load_batch(self.evalloader.dataset, self.eval_fid)
        self.construct_eval_batch(batch)
        rendered, scalars = self.model.evaluate(batch)
        self.add_image_togrid(ref_dict)
        self.add_image_togrid(rendered)
        # self.visualize_matches(rendered["xyz"], rendered["xyz_matches"], tag="xyz")
        # self.visualize_matches(
        #     rendered["xyz_cam"], rendered["xyz_reproj"], tag="xyz_cam"
        # )
        self.add_scalar(self.log, scalars, self.current_round)
        return ref_dict, rendered

    def visualize_matches(self, xyz, xyz_matches, tag):
        """Visualize dense correspondences outputted by canonical registration

        Args:
            xyz: (M,H,W,3) Predicted xyz points
            xyz_matches: (M,H,W,3) Points to match against in canonical space.
                This is an empty list for the static background model
            tag (str): Name of export mesh
        """
        if len(xyz_matches) == 0:
            return
        xyz = xyz[0].view(-1, 3).detach().cpu().numpy()
        xyz_matches = xyz_matches[0].view(-1, 3).detach().cpu().numpy()
        xyz = trimesh.Trimesh(vertices=xyz)
        xyz_matches = trimesh.Trimesh(vertices=xyz_matches)

        xyz.visual.vertex_colors = [255, 0, 0, 255]
        xyz_matches.visual.vertex_colors = [0, 255, 0, 255]
        xyz_cat = trimesh.util.concatenate([xyz, xyz_matches])

        xyz_cat.export(f"{self.save_dir}/{self.current_round:03d}-{tag}.ply")

    @staticmethod
    def load_batch(dataset, fids):
        """Load a single batch of reference frames for Tensorboard visualization

        Args:
            dataset (ConcatDataset): Eval dataset for all videos in a sequence
            fids: (nframes,) Frame indices to load
        Returns:
            ref_dict (Dict): Dict with keys "ref_rgb", "ref_mask", "ref_depth",
                "ref_feature", and "ref_flow", each (N,H,W,x)
            batch_aggr (Dict): Batch of input metadata. Keys: "dataid",
                "frameid_sub", "crop2raw", and "feature"
        """
        ref_dict = defaultdict(list)
        batch_aggr = defaultdict(list)
        ref_keys = ["rgb", "mask", "depth", "normal", "feature", "vis2d"]
        batch_keys = ["dataid", "frameid_sub", "crop2raw"]
        for fid in fids:
            batch = dataset[fid]
            for k in ref_keys:
                ref_dict[f"ref_{k}"].append(batch[k][:1])
            ref_dict["ref_flow"].append(
                batch["flow"][:1] * (batch["flow_uct"][:1] > 0).astype(float)
            )

            for k in batch_keys:
                batch_aggr[k].append(batch[k])
            batch_aggr["feature"].append(
                batch["feature"].reshape(2, -1, batch["feature"].shape[-1])
            )

        for k, v in ref_dict.items():
            ref_dict[k] = np.concatenate(v, 0)

        for k, v in batch_aggr.items():
            batch_aggr[k] = np.concatenate(v, 0)
        return ref_dict, batch_aggr

    def construct_eval_batch(self, batch):
        """Modify a batch in-place for evaluation

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid",
                "frameid_sub", "crop2raw", and "feature". This function
                modifies it in place to add key "hxy"
        """
        opts = self.opts
        # to tensor
        for k, v in batch.items():
            batch[k] = torch.tensor(v, device=self.device)

        batch["crop2raw"][..., :2] *= opts["train_res"] / opts["eval_res"]

        if not hasattr(self, "hxy"):
            hxy = self.create_xy_grid(opts["eval_res"], self.device)
            self.hxy_cache = hxy[None].expand(len(batch["dataid"]), -1, -1)
        batch["hxy"] = self.hxy_cache

    @staticmethod
    def create_xy_grid(eval_res, device):
        """Create a grid of pixel coordinates on the image plane

        Args:
            eval_res (int): Resolution to evaluate at
            device (torch.device): Target device
        Returns:
            hxy: (eval_res^2, 3) Homogeneous pixel coords on the image plane
        """
        eval_range = torch.arange(eval_res, dtype=torch.float32, device=device)
        hxy = torch.cartesian_prod(eval_range, eval_range)
        hxy = torch.stack([hxy[:, 1], hxy[:, 0], torch.ones_like(hxy[:, 0])], -1)
        return hxy

    def add_image_togrid(self, rendered_seq):
        """Add rendered outputs to Tensorboard visualization grid

        Args:
            rendered_seq (Dict): Dict of volume-rendered outputs. Keys:
                "mask" (M,H,W,1), "vis2d" (M,H,W,1), "depth" (M,H,W,1),
                "flow" (M,H,W,2), "feature" (M,H,W,16), "normal" (M,H,W,3), and
                "eikonal" (M,H,W,1)
        """
        for k, v in rendered_seq.items():
            if len(v) == 0:
                continue
            img_grid = make_image_grid(v)
            self.add_image(self.log, k, img_grid, self.current_round)

    def add_image(self, log, tag, img, step):
        """Convert volume-rendered outputs to RGB and add to Tensorboard

        Args:
            log (SummaryWriter): Tensorboard logger
            tag (str): Image tag
            img: (H_out, W_out, x) Image to show
            step (int): Current step
        """
        if len(img.shape) == 2:
            formats = "HW"
        else:
            formats = "HWC"

        img = img2color(tag, img, pca_fn=self.data_info["apply_pca_fn"])

        log.add_image("img_" + tag, img, step, dataformats=formats)

    @staticmethod
    def add_scalar(log, dict, step):
        """Add a scalar value to Tensorboard log"""
        for k, v in dict.items():
            log.add_scalar(k, v, step)

    @staticmethod
    def construct_test_model(opts, model_class=dvr_model, return_refs=True, force_reload=True):
        """Load a model at test time

        Args:
            opts (Dict): Command-line options
        """
        # io
        logname = f"{opts['seqname']}-{opts['logname']}"
        print(f"seqname:{opts['seqname']}")
        print(f"logname:{opts['logname']}")
        meta_filename = f"{opts['logroot']}/{logname}/metadata.pth"

        if not os.path.exists(meta_filename):
            force_reload = True

        # Create uncertainty map for each video
        # Initialize pixels to have uncertainty=1, which is considered high
        uncertainty_map = {}
        config = configparser.RawConfigParser()
        config.read(f"database/configs/{opts['seqname']}.config")
        for vidid in range(len(config.sections()) - 1):
            img_path = str(config.get(f"data_{vidid}", "img_path")).strip("/")
            vidname = img_path.split("/")[-1]
            uncertainty_name = f"{vidname}-{opts['data_prefix']}-{opts['train_res']}"
            rgb_path = f"{img_path}/{opts['data_prefix']}-{opts['train_res']}.npy"

            shape = np.load(rgb_path, mmap_mode="r").shape[:-1]  # N, H, W
            uncertainty = np.ones(shape, dtype=np.float32)
            uncertainty_map[uncertainty_name] = uncertainty

        # construct dataset
        if return_refs or force_reload:
            eval_dict = Trainer.construct_dataset_opts(opts, is_eval=True)
            evalloader = data_utils.eval_loader(eval_dict, uncertainty_map)
        
        if force_reload:
            data_info, _ = data_utils.get_data_info(evalloader)

            # save data_info to tmp file
            data_info_save = data_info.copy()
            del data_info_save["apply_pca_fn"]
            with open(meta_filename, "wb") as handle:
                torch.save(data_info_save, handle)
        else:
            # this will miss pca function
            with open(meta_filename, "rb") as handle:
                data_info = torch.load(handle)

        # construct DVR model
        model = model_class(opts, data_info, uncertainty_map)
        load_path = f"{opts['logroot']}/{logname}/ckpt_{opts['load_suffix']}.pth"
        _ = Trainer.load_checkpoint(load_path, model)
        model.cuda()
        model.eval()

        if "inst_id" in opts and return_refs:
            # get reference images
            inst_id = opts["inst_id"]
            offset = data_info["frame_info"]["frame_offset"]
            frame_id = np.asarray(
                range(offset[inst_id] - inst_id, offset[inst_id + 1] - inst_id - 1)
            )  # to account for pairs
            # only load a single frame
            if "freeze_id" in opts and opts["freeze_id"] > -1:
                frame_id = frame_id[opts["freeze_id"] : opts["freeze_id"] + 1]
            ref_dict, _ = Trainer.load_batch(evalloader.dataset, frame_id)
        else:
            ref_dict = None

        return model, data_info, ref_dict

    def check_grad(self, thresh=50.0):
        """Check if gradients are above a threshold

        Args:
            thresh (float): Gradient clipping threshold
        """
        opts = self.opts

        # detect large gradients and reload model
        params_list = []
        for param_dict in self.params_ref_list:
            ((name, p),) = param_dict.items()
            if p.requires_grad and p.grad is not None:
                params_list.append(p)
                # if p.grad.isnan().any():
                #     p.grad.zero_()

        # check individual parameters
        grad_norm = torch.nn.utils.clip_grad_norm_(params_list, thresh)
        if (self.current_round > 2 and grad_norm > thresh) or torch.isnan(grad_norm):
            # print diagnostics for each large gradient
            print(f"[{self.opts['seqname']}] Found large grad during training: {grad_norm:.2f}, clearing gradients")
            for param_dict in self.params_ref_list:
                ((name, p),) = param_dict.items()
                if p.requires_grad and p.grad is not None:
                    if p.grad.abs().max() > 1:
                        print(
                            f"[{self.opts['seqname']}] '{name}' grad: "
                            f"min={p.grad.min().item():.6f}, max={p.grad.max().item():.6f}, "
                            f"avg={p.grad.mean().item():.6f}, std={p.grad.std().item():.6f}"
                        )
            
            # clear gradients
            self.optimizer.zero_grad()

            # load cached model from two rounds ago
            if self.model_cache[0] is not None:
                print(f"[{self.opts['seqname']}] fallback to cached model")
                self.model.load_state_dict(self.model_cache[0])
                self.optimizer.load_state_dict(self.optimizer_cache[0])
                try:
                    self.scheduler.load_state_dict(self.scheduler_cache[0])
                except KeyError:
                    pass
            return {}

        # clip individual parameters
        grad_dict = {}
        queue_length = 10
        for param_dict in self.params_ref_list:
            ((name, p),) = param_dict.items()
            if p.requires_grad and p.grad is not None:
                grad = p.grad.reshape(-1).norm(2, -1)
                grad_dict["grad/" + name] = grad
                # maintain a queue of grad norm, and clip outlier grads
                matched_strict, clip_strict = match_param_name(
                    name, self.param_clip_startwith, type="startwith"
                )
                if matched_strict:
                    scale_threshold = clip_strict
                else:
                    continue

                # check the gradient norm
                if name not in self.grad_queue:
                    self.grad_queue[name] = []
                if len(self.grad_queue[name]) > queue_length:
                    med_grad = torch.stack(self.grad_queue[name][:-1]).median()
                    grad_dict["grad_med/" + name] = med_grad
                    if grad > scale_threshold * med_grad:
                        torch.nn.utils.clip_grad_norm_(p, med_grad)
                    else:
                        self.grad_queue[name].append(grad)
                        self.grad_queue[name].pop(0)
                else:
                    self.grad_queue[name].append(grad)

        return grad_dict
