# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
from collections import defaultdict

import configparser
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from lab4d.nnutils.intrinsics import IntrinsicsConst
from lab4d.nnutils.pose import CameraMLP_so3
from lab4d.nnutils.feature import FeatureNeRF
from lab4d.nnutils.multifields import MultiFields
from lab4d.utils.geom_utils import K2inv, K2mat
from lab4d.utils.numpy_utils import interp_wt
from lab4d.utils.render_utils import render_pixel


class dvr_model(nn.Module):
    """A model that contains a collection of static/deformable neural fields

    Args:
        config (Dict): Command-line args
        data_info (Dict): Dataset metadata from get_data_info()
        uncertainty_map (Dict(np.array)): Uncertainty map for each video
    """

    def __init__(self, config, data_info, uncertainty_map=None):
        super().__init__()
        self.config = config
        self.device = "cuda"
        self.data_info = data_info
        self.uncertainty_map = uncertainty_map

        self.fields = MultiFields(
            data_info=data_info,
            field_type=config["field_type"],
            fg_motion=config["fg_motion"],
            single_inst=config["single_inst"],
            single_scene=config["single_scene"],
        )

        intrinsics = self.data_info["intrinsics"]
        frame_info = self.data_info["frame_info"]
        self.intrinsics = IntrinsicsConst(intrinsics, frame_info)
        # self.intrinsics = IntrinsicsMLP(intrinsics, frame_info)

    def mlp_init(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics for all neural fields from external priors
        """
        self.fields.mlp_init()
        self.intrinsics.mlp_init()

    def forward(self, batch):
        """Run forward pass and compute losses

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2),
                "hxy" (M,2,N,3), and "is_detected" (M,2)
        Returns:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,).
        """
        config = self.config
        self.process_frameid(batch)
        self.reshape_batch(batch)
        results = self.render(batch, flow_thresh=config["train_res"])
        loss_dict = self.compute_loss(batch, results)
        return loss_dict

    def process_frameid(self, batch):
        """Convert frameid within each video to overall frame id

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16), and
                "hxy" (M,N,3). This function modifies it in place to add key
                "frameid" (M,)
        """
        # convert frameid_sub to frameid
        frame_offset_raw = torch.tensor(
            self.data_info["frame_info"]["frame_offset_raw"], device=batch["frameid_sub"].device
        )
        batch["frameid"] = batch["frameid_sub"] + frame_offset_raw[batch["dataid"]]

    def set_progress(self, current_steps, progress):
        """Adjust loss weights and other constants throughout training

        Args:
            current_steps (int): Number of optimization steps so far
            progress (float): Fraction of training completed (in the current stage)
        """

        self.current_steps = current_steps
        config = self.config
        if self.config["use_freq_anneal"]:
            # positional encoding annealing
            anchor_x = (1000, 2000)
            anchor_y = (0.6, 1)
            type = "linear"
            alpha = interp_wt(anchor_x, anchor_y, current_steps, type=type)
            if alpha >= 1:
                alpha = -1
            alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)
            self.fields.set_alpha(alpha)

        # # use 2k steps to warmup
        # if current_steps < 2000:
        #     self.fields.set_importance_sampling(False)
        # else:
        #     self.fields.set_importance_sampling(True)
        self.fields.set_importance_sampling(False)

        # pose correction: steps(0->2k, 1->0)
        if config["pose_correction"]:
            anchor_x = (0.8, 1.0)
            type = "linear"
            wt_modifier_dict = {
                "feat_reproj_wt": 10.0,
                "mask_wt": 0.0,
                "rgb_wt": 0.0,
                "flow_wt": 0.0,
                "feature_wt": 0.0,
                "reg_gauss_mask_wt": 0.0,
            }
            for loss_name, wt_modifier in wt_modifier_dict.items():
                anchor_y = (wt_modifier, 1.0)
                self.set_loss_weight(loss_name, anchor_x, anchor_y, progress, type=type)

        if config["pose_correction"]:
            sample_around_surface = True
        else:
            sample_around_surface = False

        if "fg" in self.fields.field_params.keys() and isinstance(
            self.fields.field_params["fg"], FeatureNeRF
        ):
            self.fields.field_params["fg"].set_match_region(sample_around_surface)

        if config["alter_flow"]:
            # alternating between flow and all losses for initialzation
            switch_list = [
                "mask_wt",
                "rgb_wt",
                "normal_wt",
                "reg_gauss_mask_wt",
            ]
            if current_steps < 1600 and current_steps % 2 == 0:
                # set to 0
                for key in switch_list:
                    self.set_loss_weight(
                        key, (0, 1), (0, 0), current_steps, type="linear"
                    )
            else:
                # set to 1x
                for key in switch_list:
                    self.set_loss_weight(
                        key, (0, 1), (1, 1), current_steps, type="linear"
                    )

        # only enable normal loss after shape is in place: steps(4k->6k, 0->1)
        loss_name = "normal_wt"
        anchor_x = (0, 2000)
        anchor_y = (0.01, 1.0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # use a higher mask weight at the start: steps(0->4k, 10->1)
        loss_name = "mask_wt"
        anchor_x = (0, 4000)
        anchor_y = (10.0, 1.0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # anneal geometry/appearance code for foreground: steps(0->2k, 1->0.2), range (0.2,1)
        anchor_x = (0, 2000)
        anchor_y = (1.0, 0.2)
        type = "linear"
        beta_prob = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        self.fields.set_beta_prob(beta_prob)

        # camera prior wt: steps(0->1000, 1->0), range (0,1)
        loss_name = "reg_cam_prior_wt"
        anchor_x = (0, config["num_rounds_cam_init"] * config["iters_per_round"] * config["grad_accum"])
        anchor_y = (1, 0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # reg_eikonal_wt: steps(0->24000, 1->100), range (1,100)
        loss_name = "reg_eikonal_wt"
        anchor_x = (800, 2000)
        anchor_y = (1, config["reg_eikonal_scale_max"])
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # skel prior wt: steps(0->4000, 1->0), to discouage large changes when shape is not good
        loss_name = "reg_skel_prior_wt"
        anchor_x = (200, 400)
        anchor_y = (10, 1)
        type = "log"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # # gauss mask wt: steps(0->4000, 1->0), range (0,1)
        # loss_name = "reg_gauss_mask_wt"
        # anchor_x = (0, 2000)
        # anchor_y = (1, 0.1)
        # type = "log"
        # self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # delta skin wt: steps(0->2000, 1->0.1), to make skinning more flexible
        loss_name = "reg_delta_skin_wt"
        anchor_x = (0, 2000)
        anchor_y = (1, 0.01)
        type = "log"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # gauss skin wt: steps(0->2000, 1->0), to align skeleton with shape
        loss_name = "reg_gauss_skin_wt"
        anchor_x = (1000, 2000)
        anchor_y = (0.05, 1)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # # learn feature field before reproj error
        # loss_name = "feat_reproj_wt"
        # anchor_x = (200, 400)
        # anchor_y = (0, 1)
        # type = "linear"
        # self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

    def set_loss_weight(
        self, loss_name, anchor_x, anchor_y, current_steps, type="linear"
    ):
        """Set a loss weight according to the current training step

        Args:
            loss_name (str): Name of loss weight to set
            anchor_x: Tuple of optimization steps [x0, x1]
            anchor_y: Tuple of loss values [y0, y1]
            current_steps (int): Current optimization step
            type (str): Interpolation type ("linear" or "log")
        """
        if f"{loss_name}_init" not in self.config.keys():
            self.config[f"{loss_name}_init"] = self.config[loss_name]
        factor = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        self.config[loss_name] = self.config[f"{loss_name}_init"] * factor

    @torch.no_grad()
    def evaluate(self, batch, is_pair=True):
        """Evaluate a Lab4D model

        Args:
            batch (Dict): Dataset metadata from `construct_eval_batch()`. Keys:
                "dataid" (M,), "frameid_sub" (M,), "crop2raw" (M,4),
                "feature" (M,N,16), and "hxy" (M,N,3)
            is_pair (bool): Whether to evaluate by rendering pairs
        Returns:
            rendered (Dict): Dict of rendered outputs. Keys: "mask" (M,H,W,1),
                "vis" (M,H,W,1), "depth" (M,H,W,1), "flow" (M,H,W,2),
                "feature" (M,H,W,16), "normal" (M,H,W,3), and
                "eikonal" (M,H,W,1)
        """
        if is_pair:
            div_factor = 2
        else:
            div_factor = 1
        self.process_frameid(batch)

        rendered = defaultdict(list)
        # split batch
        for i in trange(0, len(batch["frameid"]) // div_factor, desc=f"Rendering frames"):
            batch_sub = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    batch_sub[k] = {}
                    for k2, v2 in v.items():
                        batch_sub[k][k2] = v2[i * div_factor : (i + 1) * div_factor]
                else:
                    batch_sub[k] = v[i * div_factor : (i + 1) * div_factor]
            results_sub = self.render(batch_sub)
            rendered_sub, aux = results_sub["rendered"], results_sub["aux_dict"]
            for k, v in rendered_sub.items():
                res = int(np.sqrt(v.shape[1]))
                out = v.view(div_factor, res, res, -1)[0].detach().cpu()
                rendered[k].append(out)
            # for k, v in aux["fg"].items():
            #     res = int(np.sqrt(v.shape[1]))
            #     out = v.view(div_factor, res, res, -1)[0].detach().cpu()
            #     rendered[f"{k}_id-fg"].append(out)
            # for k, v in aux["bg"].items():
            #     res = int(np.sqrt(v.shape[1]))
            #     out = v.view(div_factor, res, res, -1)[0].detach().cpu()
            #     rendered[f"{k}_id-bg"].append(out)
        # Debug print available keys
        print("Available rendered keys:", rendered.keys())

        # First, stack all rendered items into tensors and verify
        for k, v in rendered.items():
            try:
                rendered[k] = torch.stack(v, 0)
            except Exception as e:
                print(f"Error stacking tensor for key {k}: {e}")
                print(f"Shape/type of v: {type(v)}, len: {len(v)}")
                raise

        # Check for required mask and provide fallback
        if "mask_id-fg" not in rendered:
            if "mask" in rendered:
                rendered["mask_id-fg"] = rendered["mask"]
            else:
                # Create a default mask of ones if no mask is available
                sample_key = next(iter(rendered.keys()))
                mask_shape = rendered[sample_key].shape[:3] + (1,)  # Assuming BHWC format
                rendered["mask_id-fg"] = torch.ones(mask_shape, device=rendered[sample_key].device)

        # Then apply masks with verification
        for k, v in rendered.items():
            if "mask" in k:
                continue
            elif "xyz_matches" in k or "xyz_reproj" in k:
                mask = rendered["mask_id-fg"]
                if not isinstance(mask, torch.Tensor):
                    raise TypeError(f"mask_id-fg is {type(mask)}, expected torch.Tensor")
                rendered[k] = rendered[k] * (mask > 0.5).float()
            elif "xy_reproj" in k:
                mask = batch["feature"][::div_factor].norm(2, -1, keepdim=True) > 0
                res = rendered[k].shape[1]
                rendered[k] = rendered[k] * mask.float().view(-1, res, res, 1).detach().cpu()
            else:
                if "id-fg" in k:
                    mask = rendered["mask_id-fg"]
                elif "id-bg" in k:
                    mask = rendered["mask_id-bg"]
                else:
                    mask = rendered["mask"]
                rendered[k] = rendered[k] * mask

        scalars = {}
        return rendered, scalars

    def update_geometry_aux(self):
        """Extract proxy geometry for all neural fields"""
        self.fields.update_geometry_aux()

    def update_camera_aux(self):
        # update camera mlp base quat
        for field in self.fields.field_params.values():
            if isinstance(field.camera_mlp, CameraMLP_so3):
                field.camera_mlp.update_base_quat()

    def export_geometry_aux(self, path):
        """Export proxy geometry for all neural fields"""
        return self.fields.export_geometry_aux(path)

    def render(self, batch, flow_thresh=None):
        """Render model outputs

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16),
                "hxy" (M,N,3), and "frameid" (M,)
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
            results["rendered"]: "mask" (M,N,1), "rgb" (M,N,3),
                "vis" (M,N,1), "depth" (M,N,1), "flow" (M,N,2),
                "feature" (M,N,16), "normal" (M,N,3), and "eikonal" (M,N,1)
            results["aux_dict"]["fg"]: "xy_reproj" (M,N,2) and "feature" (M,N,16)
        """
        samples_dict = self.get_samples(batch)
        if self.training:
            results = self.render_samples(samples_dict, flow_thresh=flow_thresh)
        else:
            results = self.render_samples_chunk(samples_dict, flow_thresh=flow_thresh)
        return results

    def get_samples(self, batch):
        """Compute time-dependent camera and articulation parameters for all
        neural fields.

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16),
                "hxy" (M,N,3), and "frameid" (M,)
        Returns:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,4,4), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,3), and
                "feature" (M,N,16).
        """
        if "Kinv" in batch.keys():
            Kinv = batch["Kinv"]
        else:
            Kmat = self.intrinsics.get_vals(batch["frameid"])
            Kinv = K2inv(Kmat) @ K2mat(batch["crop2raw"])

        samples_dict = self.fields.get_samples(Kinv, batch)
        return samples_dict

    def render_samples_chunk(self, samples_dict, flow_thresh=None, chunk_size=2048):
        """Render outputs from all neural fields. Divide in chunks along pixel
        dimension N to avoid running out of memory.

        Args:
            samples_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts of input metadata and time-dependent outputs.
                Each dict has keys: "Kinv" (M,3,3), "field2cam" (M,4,4),
                "frame_id" (M,), "inst_id" (M,), "near_far" (M,2),
                "hxy" (M,N,3), and "feature" (M,N,16).
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
            chunk_size (int): Number of pixels to render per chunk
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        # get chunk size
        category = list(samples_dict.keys())[0]
        num_imgs, num_pixels, _ = samples_dict[category]["hxy"].shape
        total_pixels = num_imgs * num_pixels
        num_chunks = int(np.ceil(total_pixels / chunk_size))

        # break into chunks at pixel dimension
        chunk_size_px = int(np.ceil(chunk_size // num_imgs))

        results = {
            "rendered": defaultdict(list),
            "aux_dict": defaultdict(defaultdict),
        }
        for i in range(num_chunks):
            # construct chunk input
            samples_dict_chunk = defaultdict(list)
            for category, category_v in samples_dict.items():
                samples_dict_chunk[category] = defaultdict(list)
                for k, v in category_v.items():
                    # only break for pixel-ish elements
                    if (
                        isinstance(v, torch.Tensor)
                        and v.ndim == 3
                        and v.shape[1] == num_pixels
                    ):
                        chunk_px = v[:, i * chunk_size_px : (i + 1) * chunk_size_px]
                        samples_dict_chunk[category][k] = chunk_px.clone()
                    else:
                        samples_dict_chunk[category][k] = v

            # get chunk output
            if not self.training:
                # clear cache for evaluation
                torch.cuda.empty_cache()
            results_chunk = self.render_samples(
                samples_dict_chunk, flow_thresh=flow_thresh
            )

            # merge chunk output
            for k, v in results_chunk["rendered"].items():
                if k not in results["rendered"].keys():
                    results["rendered"][k] = []
                results["rendered"][k].append(v)

            for cate in results_chunk["aux_dict"].keys():
                for k, v in results_chunk["aux_dict"][cate].items():
                    if k not in results["aux_dict"][cate].keys():
                        results["aux_dict"][cate][k] = []
                    results["aux_dict"][cate][k].append(v)

        # concat chunk output
        for k, v in results["rendered"].items():
            results["rendered"][k] = torch.cat(v, 1)

        for cate in results["aux_dict"].keys():
            for k, v in results["aux_dict"][cate].items():
                results["aux_dict"][cate][k] = torch.cat(v, 1)
        return results

    def render_samples(self, samples_dict, flow_thresh=None):
        """Render outputs from all neural fields.

        Args:
            samples_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts of input metadata and time-dependent outputs.
                Each dict has keys: "Kinv" (M,3,3), "field2cam" (M,4,4),
                "frame_id" (M,), "inst_id" (M,), "near_far" (M,2),
                "hxy" (M,N,3), and "feature" (M,N,16).
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        multifields_dict, deltas_dict, aux_dict = self.fields.query_multifields(
            samples_dict, flow_thresh=flow_thresh
        )

        field_dict, deltas = self.fields.compose_fields(multifields_dict, deltas_dict)
        rendered = render_pixel(field_dict, deltas)

        for cate in multifields_dict.keys():
            # render each field and put into aux_dict
            rendered_cate = render_pixel(multifields_dict[cate], deltas_dict[cate])
            for k, v in rendered_cate.items():
                aux_dict[cate][k] = v

        if "fg" in aux_dict.keys():
            # move for visualization
            if "xyz_matches" in aux_dict["fg"].keys():
                rendered["xyz_matches"] = aux_dict["fg"]["xyz_matches"]
                rendered["xyz_reproj"] = aux_dict["fg"]["xyz_reproj"]

        results = {"rendered": rendered, "aux_dict": aux_dict}
        return results

    @staticmethod
    def reshape_batch(batch):
        """Reshape a batch to merge the pair dimension into the batch dimension

        Args:
            batch (Dict): Arbitrary dataloader outputs (M, 2, ...). This is
                modified in place to reshape each value to (M*2, ...)
        """
        for k, v in batch.items():
            batch[k] = v.view(-1, *v.shape[2:])

    def compute_loss(self, batch, results):
        """Compute model losses

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        Returns:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,).
        """
        config = self.config
        loss_dict = {}
        self.compute_recon_loss(loss_dict, results, batch, config, self.current_steps)
        self.mask_losses(loss_dict, batch, config)
        self.update_uncertainty(loss_dict, batch, config)
        self.compute_reg_loss(loss_dict, results)
        motion_scale = torch.tensor(self.data_info["motion_scales"], device=self.device)
        motion_scale = motion_scale[batch["dataid"]]
        self.apply_loss_weights(loss_dict, config, motion_scale)
        return loss_dict

    @staticmethod
    def get_mask_balance_wt(mask, vis2d, is_detected):
        """Balance contribution of positive and negative pixels in mask.

        Args:
            mask: (M,N,1) Object segmentation mask
            vis2d: (M,N,1) Whether each pixel is visible in the video frame
            is_detected: (M,) Whether there is segmentation mask in the frame
        Returns:
            mask_balance_wt: (M,N,1) Balanced mask
        """
        # all the positive labels
        mask = mask.float()
        # all the labels
        vis2d = vis2d.float() * is_detected.float()[:, None, None]
        if mask.sum() > 0 and (1 - mask).sum() > 0:
            pos_wt = vis2d.sum() / mask[vis2d > 0].sum()
            neg_wt = vis2d.sum() / (1 - mask[vis2d > 0]).sum()
            mask_balance_wt = 0.5 * pos_wt * mask + 0.5 * neg_wt * (1 - mask)
        else:
            mask_balance_wt = 1
        return mask_balance_wt

    @staticmethod
    def compute_recon_loss(loss_dict, results, batch, config, current_steps):
        """Compute reconstruction losses.

        Args:
            loss_dict (Dict): Updated in place to add keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1), and
                "reg_gauss_mask" (M,N,1)
            results: Rendered outputs. Keys: "rendered", "aux_dict"
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            config (Dict): Command-line options
        """
        rendered = results["rendered"]
        aux_dict = results["aux_dict"]
        # reconstruction loss
        # get rendered fg mask
        if config["field_type"] == "fg":
            rendered_fg_mask = rendered["mask"]
        elif config["field_type"] == "comp":
            rendered_fg_mask = rendered["mask_fg"]
            # rendered_fg_mask = aux_dict["fg"]["mask"]
        elif config["field_type"] == "bg":
            rendered_fg_mask = None
        else:
            raise ValueError(f"field_type {config['field_type']} not supported")
        # get fg mask balance factor
        mask_balance_wt = dvr_model.get_mask_balance_wt(
            batch["mask"], batch["vis2d"], batch["is_detected"]
        )
        if config["field_type"] == "bg":
            loss_dict["mask"] = (rendered["mask"] - 1).pow(2)
        elif config["field_type"] == "fg":
            loss_dict["mask"] = (rendered_fg_mask - batch["mask"].float()).pow(2)
            loss_dict["mask"] *= mask_balance_wt
        elif config["field_type"] == "comp":
            loss_dict["mask"] = (rendered["mask"] - 1).pow(2)
        else:
            raise ValueError(f"field_type {config['field_type']} not supported")

        if config["field_type"] == "fg" or config["field_type"] == "comp":
            loss_dict["feature"] = (aux_dict["fg"]["feature"] - batch["feature"]).norm(
                2, -1, keepdim=True
            )
            loss_dict["feat_reproj"] = aux_dict["fg"]["xy_reproj"]

        loss_dict["rgb"] = (rendered["rgb"] - batch["rgb"]).pow(2)
        loss_dict["depth"] = (rendered["depth"] - batch["depth"]).abs()
        # remove pixels not sampled to render normals
        loss_dict["normal"] = (
            torch.linalg.norm(batch["normal"], dim=-1, keepdims=True) *
            torch.linalg.norm(rendered["normal"], dim=-1, keepdims=True) *
            torch.abs(rendered["normal"] - batch["normal"]).pow(2)
        )
        loss_dict["flow"] = (rendered["flow"] - batch["flow"]).norm(2, -1, keepdim=True)

        # visibility: supervise on fg and bg separately
        vis_loss = []
        # for aux_cate_dict in aux_dict.values():
        for cate, aux_cate_dict in aux_dict.items():
            if cate == "bg":
                # use smaller weight for bg
                aux_cate_dict["vis"] *= 0.01
            vis_loss.append(aux_cate_dict["vis"])
        vis_loss = torch.stack(vis_loss, 0).sum(0)
        loss_dict["vis"] = vis_loss

        # weighting
        loss_dict["flow"] = loss_dict["flow"] * (batch["flow_uct"] > 0).float()

        # consistency between rendered mask and gauss mask
        if "gauss_mask" in rendered.keys():
            if current_steps < 4000:
                # supervise with a fixed target
                loss_dict["reg_gauss_mask"] = (
                    aux_dict["fg"]["gauss_mask"] - batch["mask"].float()
                ).pow(2)
            else:
                loss_dict["reg_gauss_mask"] = (
                    aux_dict["fg"]["gauss_mask"] - (rendered_fg_mask > 0.5).float()
                ).pow(2)

        # # downweight pixels with low opacity (e.g., mask not aligned with gt)
        # density_related_loss = ["rgb", "depth", "normal", "feature", "flow"]
        # for k in density_related_loss:
        #     loss_dict[k] = loss_dict[k] * rendered["mask"].detach()

    @torch.no_grad()
    def update_uncertainty(self, loss_dict, batch, config):
        """Update uncertainty with per-pixel loss using exponential moving average

        Args:
            loss_dict (Dict): Dict with keys: "mask" (M,N,1), "rgb" (M,N,3), and others
            batch (Dict): Batch of dataloader samples
            config (Dict): Command-line options
        """
        if self.uncertainty_map is None:
            return
        for i, k_unc in enumerate(self.uncertainty_map):
            valid_mask = (batch["dataid"] == i)
            if valid_mask.sum() == 0:
                continue
        
            # Compute index of sampled pixels
            xs = batch["hxy"][:, :, 0].to(torch.int64).detach().cpu().numpy()  # M, N
            ys = batch["hxy"][:, :, 1].to(torch.int64).detach().cpu().numpy()  # M, N
            frameid = batch["frameid"][:, None].expand(-1, xs.shape[-1]).detach().cpu().numpy()  # M, N

            # Compute uncertainty for relevant keys
            unc = None
            unc_keys = ["rgb", "mask"]
            wt_sum = 0
            for k in unc_keys:
                assert loss_dict[k].ndim == 3 and loss_dict[k].shape[:2] == xs.shape, (loss_dict[k].shape, xs.shape)
                unc_val_k = config[f"{k}_wt"] * torch.norm(loss_dict[k], dim=-1)  # M, N
                wt_sum += config[f"{k}_wt"]
                if unc is None:
                    unc = unc_val_k
                else:
                    unc += unc_val_k
            if unc is None:
                continue
            unc /= wt_sum

            # Update uncertainty with exponential moving average
            alpha = config["uncertainty_ema"]
            curr_unc = self.uncertainty_map[k_unc][frameid, ys, xs]
            new_unc = alpha * curr_unc + (1 - alpha) * unc.detach().cpu().numpy()
            self.uncertainty_map[k_unc][frameid, ys, xs] = new_unc

    def compute_reg_loss(self, loss_dict, results):
        """Compute regularization losses.

        Args:
            loss_dict (Dict): Updated in place to add keys:
                "reg_visibility" (0,), "reg_eikonal" (0,),
                "reg_deform_cyc" (0,), "reg_soft_deform" (0,),
                "reg_gauss_skin" (0,), "reg_cam_prior" (0,), and
                "reg_skel_prior" (0,).
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        rendered = results["rendered"]
        aux_dict = results["aux_dict"]
        # regularization loss
        loss_dict["reg_visibility"] = self.fields.visibility_decay_loss()
        loss_dict["reg_eikonal"] = rendered["eikonal"]
        if "fg" in aux_dict.keys():
            loss_dict["reg_deform_cyc"] = aux_dict["fg"]["cyc_dist"]
            loss_dict["reg_delta_skin"] = aux_dict["fg"]["delta_skin"]
            loss_dict["reg_skin_entropy"] = aux_dict["fg"]["skin_entropy"]
        loss_dict["reg_soft_deform"] = self.fields.soft_deform_loss()
        if self.config["reg_gauss_skin_wt"] > 0:
            loss_dict["reg_gauss_skin"] = self.fields.gauss_skin_consistency_loss()
        loss_dict["reg_cam_prior"] = self.fields.cam_prior_loss()
        loss_dict["reg_skel_prior"] = self.fields.skel_prior_loss()
        loss_dict["reg_joint_prior"] = self.fields.joint_prior_loss()

    @staticmethod
    def mask_losses(loss_dict, batch, config):
        """Apply segmentation mask on dense losses

        Args:
            loss_dict (Dict): Dense losses. Keys: "mask" (M,N,1), "rgb" (M,N,3),
                "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1), "feature" (M,N,1),
                "feat_reproj" (M,N,1), and "reg_gauss_mask" (M,N,1). Modified in
                place to multiply loss_dict["mask"] with the other losses
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            config (Dict): Command-line options
        """
        # ignore the masking step
        keys_ignore_masking = ["reg_gauss_mask"]
        # always mask-out non-visible (out-of-frame) pixels
        keys_allpix = ["mask"]
        # always mask-out non-object pixels
        keys_fg = ["feature", "feat_reproj"]
        # field type specific keys
        keys_type_specific = ["rgb", "depth", "normal", "flow", "vis"]

        # type-specific masking rules
        vis2d = batch["vis2d"].float()
        maskfg = batch["mask"].float()
        if config["field_type"] == "bg":
            mask = (1 - maskfg) * vis2d
        elif config["field_type"] == "fg":
            mask = maskfg * vis2d
        elif config["field_type"] == "comp":
            mask = vis2d
        else:
            raise ValueError(f"field_type {config['field_type']} not supported")

        # apply mask
        for k, v in loss_dict.items():
            if k in keys_ignore_masking:
                continue
            elif k in keys_allpix:
                loss_dict[k] = v * vis2d
            elif k in keys_fg:
                loss_dict[k] = v * maskfg
            elif k in keys_type_specific:
                loss_dict[k] = v * mask
            else:
                raise ValueError(f"loss {k} not defined")

        # mask out the following losses if obj is not detected
        keys_mask_not_detected = ["feature", "feat_reproj"]
        is_detected = batch["is_detected"].float()[:, None, None]
        for k, v in loss_dict.items():
            if k in keys_mask_not_detected:
                loss_dict[k] = v * is_detected

        # remove mask loss for frames without detection
        if config["field_type"] == "fg" or config["field_type"] == "comp":
            loss_dict["mask"] = loss_dict["mask"] * is_detected

    @staticmethod
    def apply_loss_weights(loss_dict, config, motion_scale):
        """Weigh each loss term according to command-line configs

        Args:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,). Modified in
                place to multiply each term with a scalar weight.
            config (Dict): Command-line options
            motion_scale (Tensor): Motion magnitude for each data sample (M,)
        """
        # px_unit_keys = ["feat_reproj"]
        # motion_unit_keys = ["flow"]
        px_unit_keys = ["feat_reproj", "flow"]
        for k, v in loss_dict.items():
            # # scale with motion magnitude
            # if k in motion_unit_keys:
            #     loss_dict[k] /= motion_scale.clamp(1, 20).view(-1, 1, 1)

            # average over non-zero pixels
            v = v[v > 0]
            if v.numel() > 0:
                loss_dict[k] = v.mean()
            else:
                loss_dict[k] = v.sum()  # return zero

            # scale with image resolution
            if k in px_unit_keys:
                loss_dict[k] /= config["train_res"]

            # scale with loss weights
            wt_name = k + "_wt"
            if wt_name in config.keys():
                loss_dict[k] *= config[wt_name]

    def get_field_betas(self):
        """Get beta values for all neural fields

        Returns:
            betas (Dict): Beta values for each neural field
        """
        beta_dicts = {}
        for field in self.fields.field_params.values():
            beta_dicts[f"beta/{field.category}"] = field.logibeta.exp()
        return beta_dicts

    @torch.no_grad()
    def get_cameras(self, frame_id=None):
        """Compute camera matrices in world units

        Returns:
            field2cam (Dict): Maps field names ("fg" or "bg") to (M,4,4) cameras
        """
        return self.fields.get_cameras(frame_id=frame_id)

    @torch.no_grad()
    def get_intrinsics(self, inst_id=None):
        return self.intrinsics.get_intrinsics(inst_id=inst_id)

    @torch.no_grad()
    def get_aabb(self, inst_id=None):
        """Compute axis aligned bounding box
        Args:
            inst_id (int or tensor): Instance id. If None, return aabb for all instances

        Returns:
            aabb (Dict): Maps field names ("fg" or "bg") to (1/N,2,3) aabb
        """
        return self.fields.get_aabb(inst_id=inst_id)
