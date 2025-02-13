import os, sys
import numpy as np
import torch
import time
from torch import nn
import torch.nn.functional as F
import pdb
import cv2
import tqdm
import math
import trimesh

from gsplat import rasterization

from lab4d.config import load_flags_from_file
from lab4d.engine.trainer import Trainer
from lab4d.engine.model import dvr_model
from lab4d.nnutils.intrinsics import IntrinsicsConst
from lab4d.utils.numpy_utils import interp_wt
from lab4d.utils.loss_utils import get_mask_balance_wt
from lab4d.utils.geom_utils import (
    K2mat,
    K2inv,
    pinhole_projection,
    rot_angle,
)
from lab4d.utils.camera_utils import get_rotating_cam
from lab4d.utils.quat_transform import (
    quaternion_mul,
    matrix_to_quaternion,
    quaternion_translation_to_se3,
)
from lab4d.dataloader import data_utils
from lab4d.diffgs.gs_renderer import GaussianModel, gs_transform
from lab4d.diffgs.sh_utils import eval_sh, SH2RGB, RGB2SH
from lab4d.diffgs.cam_utils import orbit_camera

from lab4d.utils.flow_utils import point_vec, warp_flow

def load_lab4d(config):
    flags_path = config["lab4d_path"]
    # load lab4d model
    if len(flags_path) == 0:
        _, data_info, _ = Trainer.construct_test_model(config, return_refs=False, force_reload=False)
        model = dvr_model(config, data_info)
        model.cuda()
        model.eval()
        meshes = {}
        for cate, field in model.fields.field_params.items():
            meshes[cate] = field.proxy_geometry
            # physical scale of the object / 0.5m
            scale_ratio = config["gaussian_obj_scale"] / 0.5
            meshes[cate] = meshes[cate].apply_scale(scale_ratio)
    else:
        opts = load_flags_from_file(flags_path)
        opts["load_suffix"] = "latest"
        model, data_info, _ = Trainer.construct_test_model(opts, return_refs=False, force_reload=False)
        meshes = model.fields.extract_canonical_meshes(grid_size=256, vis_thresh=-10)
    # color
    for cate, field in model.fields.field_params.items():
        color = field.extract_canonical_color(meshes[cate])
        meshes[cate].visual.vertex_colors[:,:3] = color * 255
    model = model.cpu()
    return model, meshes

def fake_a_pair(tensor):
    """Fake a pair of tensors by repeating the first dimension

    Args:
        tensor (torch.Tensor): Tensor with shape (M, ...)

    Returns:
        tensor (torch.Tensor): Tensor with shape (M*2, ...)
    """
    if torch.is_tensor(tensor):
        return tensor[:, None].repeat((1, 2) + tuple([1] * (tensor.ndim - 1)))
    elif isinstance(tensor, dict):
        for k, v in tensor.items():
            tensor[k] = fake_a_pair(v)
        return tensor
    else:
        raise NotImplementedError


class GSplatModel(nn.Module):
    def __init__(self, config, data_info, uncertainty_map=None):
        super().__init__()
        self.config = config
        self.device = "cuda"
        self.data_info = data_info
        self.progress = 0.0

        # dataset info
        frame_info = data_info["frame_info"]
        frame_offset = data_info["frame_info"]["frame_offset"]
        frame_offset_raw = data_info["frame_info"]["frame_offset_raw"]
        self.frame_offset = frame_offset
        self.frame_offset_raw = frame_offset_raw
        self.frame_mapping = frame_info["frame_mapping"]
        self.num_frames = frame_offset[-1]

        # 3DGS
        sh_degree = config["sh_degree"] if "sh_degree" in config else 3

        self.sh_degree = sh_degree
        self.active_sh_degree = 0

        if config["field_type"] == "comp":
            parent_list = [-1, 0, 0]
            mode_list = ["", "bg", "fg"]
        elif config["field_type"] == "bg":
            parent_list = [-1]
            mode_list = ["bg"]
        elif config["field_type"] == "fg":
            parent_list = [-1]
            mode_list = ["fg"]
        else:
            raise NotImplementedError
        lab4d_model, lab4d_meshes = load_lab4d(config)
        self.gaussians = GaussianModel(sh_degree, config, data_info, 
                                       parent_list=parent_list, index=0, mode_list=mode_list,
                                       lab4d_model=lab4d_model, lab4d_meshes=lab4d_meshes)

        self.init_background(config["train_res"])


        # intrinsics
        self.construct_intrinsics(lab4d_model.intrinsics.get_vals().detach().cpu().numpy())

        if config["use_gui"]:
            from lab4d.diffgs.viserviewer import ViserViewer
            gui = ViserViewer(device=self.device, viewer_port=6789, data_info=data_info)
            gui.set_renderer(self)
            self.gui = gui

    def init_background(self, resolution):
        bg_color = torch.ones((3, resolution, resolution), dtype=torch.float)
        self.bg_color = nn.Parameter(bg_color)

    def get_bg_color(self):
        return self.bg_color

    def construct_intrinsics(self, intrinsics):
        """Construct camera intrinsics module"""
        config = self.config
        if config["intrinsics_type"] == "const":
            self.intrinsics = IntrinsicsConst(
                intrinsics,
                frame_info=self.data_info["frame_info"],
            )
        else:
            raise NotImplementedError

    @staticmethod
    def get_screenspace_pts_placeholder(gaussians):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros(
                (gaussians.get_num_pts, 3),
                dtype=torch.float32,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass
        return screenspace_points


    def render_pair(
        self,
        res,
        Kmat,
        w2c,
        bg_color=None,
        frameid=None,
        device=None,
    ):
        """Render a batch of view pairs given batch with shape (M,2,...)
        Args:
            res (Int): Render resolution
            w2c (torch.Tensor): World to camera matrix
            bg_color (torch.Tensor): Background color
            frameid (torch.Tensor): Frame id
        """
        chunk_size = 64
        bs = frameid.shape[0]

        # bs,... -> bs * 2
        out_dict_all = {}
        pts_dict_all = {}
        for idx in range(0, bs, chunk_size):
            Kmat_sub = Kmat[idx: idx + chunk_size]
            w2c_sub = w2c[idx: idx + chunk_size]
            frameid_sub = frameid[idx: idx + chunk_size]

            Kmat_sub = Kmat_sub.view(-1, 3, 3)
            w2c_sub = w2c_sub.view(-1, 4, 4)
            frameid_sub = frameid_sub.view(-1)

            out_dict, pts_dict = self.render(
                res,
                Kmat_sub,
                w2c_sub,
                bg_color,
                frameid_sub,
                render_flow=True,
            )
            for k, v in out_dict.items():
                v = v.view((-1, 2) + v.shape[1:])
                if device is not None:
                    v = v.to(device)
                if k in out_dict_all:
                    out_dict_all[k] = torch.cat([out_dict_all[k], v], 0)
                else:
                    out_dict_all[k] = v
            
            for k,v in pts_dict.items():
                if device is not None:
                    v = v.to(device)
                if k in pts_dict_all:
                    pts_dict_all[k] = torch.cat([pts_dict_all[k], v], 1)
                else:
                    pts_dict_all[k] = v
        return out_dict_all, pts_dict_all

    @staticmethod
    def gsplat_render(means3D, scales, quats, viewmat, Kmat, res, feat_dict, opacities, bg_color):
        Kmat_img = Kmat.clone()
        Kmat_img[...,0,0] = Kmat_img[...,0,0] * res/2
        Kmat_img[...,1,1] = Kmat_img[...,1,1] * res/2
        Kmat_img[...,0,2] = Kmat_img[...,0,2] * res/2 + res / 2
        Kmat_img[...,1,2] = Kmat_img[...,1,2] * res/2 + res / 2
        quats / quats.norm(dim=-1, keepdim=True)
        bs = means3D.shape[1]
        
        # merge features for N,bs,K
        feats = []
        out_dim = {}
        for k,v in feat_dict.items():
            out_dim[k] = v.shape[-1]
            # batched
            if v.ndim==3:
                pass
            elif v.ndim==2:
                v = v[:,None].repeat(1,bs,1) # N,bs,3
            else:
                raise ValueError
            feats.append(v)
        feats = torch.cat(feats, -1)

        rendered_images, rendered_alphas, meta = rasterization(
            means=means3D, # [N, bs, 3]
            quats=quats.view(-1,4), # [N*bs, 4]
            scales=scales, # [N, 3]
            opacities=opacities[:,0], # [N]
            colors=feats.transpose(0,1), # [bs, N, 3]
            viewmats=viewmat, # [1, 4, 4]
            Ks=Kmat_img, # [1, 3, 3]
            width=res,
            height=res,
            render_mode="RGB+D",
            backgrounds=bg_color,
            packed=False,
        )

        # merge
        rendered_images = rendered_images.permute(0,3,1,2)
        rendered_depths = rendered_images[:, -1:]
        rendered_images = rendered_images[:, :-1]
        rendered_alphas = rendered_alphas[:, None, ..., 0]

        out_dict = {}
        for k,v in out_dim.items():
            out_dict[k] = rendered_images[:, :v]
            rendered_images = rendered_images[:, v:]
        return out_dict, rendered_depths, rendered_alphas

    def render(
        self,
        res,
        Kmat,
        w2c,
        bg_color=None,
        frameid=None,
        render_flow=False,
    ):
        opacity = self.gaussians.get_opacity
        scales = self.gaussians.get_scaling
        means3D = self.gaussians.get_xyz(frameid)
        rotations = self.gaussians.get_rotation(frameid)
        shs = self.gaussians.get_colors(frameid)

        means3D, rotations = gs_transform(means3D, rotations, w2c)
        xy_1 = pinhole_projection(Kmat, means3D.transpose(0,1)).transpose(0,1)[:,:,:2]   
        
        identity_viewmat = torch.eye(4, dtype=torch.float32, device="cuda")
        bs = frameid.shape[0]
        identity_viewmat = identity_viewmat[None].repeat(bs, 1, 1)
        feature_dict = {
                        "rgb": SH2RGB(shs[...,0,:]),
                        "xyz": self.gaussians.get_xyz().detach(),
                        "feat": self.gaussians.get_features,
                        "vis2d": self.gaussians.get_vis_ratio[...,None].float(),
                        }
        
        # render flow
        if render_flow:
            means3D_2 = means3D.view(-1,bs//2,2, 3).flip(2).view(-1,bs,3)
            xy_2 = xy_1.view(-1,bs//2,2,2).flip(2).view(-1,bs,2)
            # flow filtering
            flow = (xy_2 - xy_1)
            invalid_pts = torch.logical_or(means3D[...,2] < 0.01, means3D_2[...,2] < 0.01)
            flow[invalid_pts] = 0
            # clip large flow
            flow = flow.clamp(-1, 1)
            feature_dict["flow"] = flow  * res / 2

        rendered_image, rendered_depth, rendered_alpha = self.gsplat_render(
            means3D, scales, rotations, identity_viewmat, Kmat, res,
            feature_dict, opacity, bg_color,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        out_dict = {
            "rgb": rendered_image["rgb"].clamp(0, 1),
            "depth": rendered_depth,
            "feature": rendered_image["feat"],
            "feature_fg": rendered_image["feat"],
            "alpha": rendered_alpha,
            "xyz": rendered_image["xyz"],
            "vis2d": rendered_image["vis2d"],
        }
        if "flow" in feature_dict:
            out_dict["flow"] = rendered_image["flow"]

        if self.config["field_type"] == "comp":
            # render mask_fg
            gaussians_fg = self.gaussians.gaussians[1]
            means3D_fg = gaussians_fg.get_xyz(frameid)
            opacity_fg = gaussians_fg.get_opacity
            scales_fg = gaussians_fg.get_scaling
            rotations_fg = gaussians_fg.get_rotation(frameid)

            w2c_fg = gaussians_fg.get_extrinsics(frameid)
            means3D_fg, rotations_fg = gs_transform(means3D_fg, rotations_fg, w2c_fg)

            feature_dict = {"xyz": means3D_fg}
            _, _, rendered_alpha_fg = self.gsplat_render(
                means3D_fg, scales_fg, rotations_fg, identity_viewmat, Kmat, res,
                feature_dict, opacity_fg, bg_color
            )
            out_dict["mask_fg"] = rendered_alpha_fg

        pts_dict = {"xy_1": xy_1}
        return out_dict, pts_dict


    def get_default_cam(self, render_resolution):
        # focal=1 corresponds to 90 degree fov and identity projection matrix
        # convert this to a dict
        near = 0.01
        far = 100
        w2c = np.eye(4, dtype=np.float32)
        cam_dict = {
            "w2c": w2c,
            "render_resolution": render_resolution,
            "near": near,
            "far": far,
        }
        return cam_dict

    def compute_render_Kmat(self, crop_size, crop2raw, Kmat):
        # Kmat = Kmat_crop2raw^-1 @ Kmat_raw
        if not torch.is_tensor(Kmat):
            Kmat = torch.tensor(Kmat, dtype=torch.float32, device=crop2raw.device)
        Kmat = K2inv(crop2raw) @ Kmat  # THIS IS IN THE CROP SPACE (256x256 patch)
        # normlalize Kmat from pixel to screen space
        Kmat[..., :2, 2] = Kmat[..., :2, 2] - crop_size / 2
        Kmat[..., :2, :] = Kmat[..., :2, :] / crop_size * 2
        return Kmat

    def compute_camera_samples(self, batch, crop_size):
        """Compute camera extrinsics and intrinsics
        Args:
            batch (Dict): Items with shape (M, 2, ...)
        """
        frameid_abs = batch["frameid"]
        if "crop2raw" in batch.keys():
            crop2raw = batch["crop2raw"]
        else:
            crop2raw = torch.tensor([1.0, 1.0, 0.0, 0.0], device=self.device)
            crop2raw = crop2raw[None, None].repeat(frameid_abs.shape[0], 2, 1)
        if "Kinv" in batch.keys():
            Kmat_raw = batch["Kinv"].inverse()
        else:
            shape = frameid_abs.shape
            Kmat_raw = K2mat(self.intrinsics.get_vals(frameid_abs.view(-1)).view(*shape, 4))
        Kmat_unit = self.compute_render_Kmat(crop_size, crop2raw, Kmat_raw)
        if "field2cam" in batch:
            w2c = batch["field2cam"]["fg"]
            w2c = quaternion_translation_to_se3(w2c[..., :4], w2c[..., 4:])
        else:
            w2c = self.gaussians.get_extrinsics(frameid_abs)
        return Kmat_unit, w2c

    def compute_recon_losses(self, loss_dict, rendered, batch):
        # reference view loss
        config = self.config
        loss_dict["rgb"] = (rendered["rgb"] - batch["rgb"]).pow(2)
        if config["field_type"]=="bg":
            loss_dict["mask"] = (rendered["alpha"] - 1).pow(2)
        elif config["field_type"]=="fg":
            loss_dict["mask"] = (rendered["alpha"] - batch["mask"].float()).pow(2)
        elif config["field_type"]=="comp":
            rendered_fg_mask = rendered["mask_fg"]
            loss_dict["mask"] = (rendered_fg_mask - batch["mask"].float()).pow(2)
            loss_dict["mask"] += (rendered["alpha"] - 1).pow(2)
        loss_dict["flow"] = (rendered["flow"] - batch["flow"]).norm(2, 1, keepdim=True)
        loss_dict["flow"] = loss_dict["flow"] * (batch["flow_uct"] > 0).float()
        loss_dict["depth"] = (rendered["depth"] - batch["depth"]).abs()
        loss_dict["depth"] = loss_dict["depth"] * (batch["depth"] > 0).float()
        
        if config["field_type"] == "fg" or config["field_type"] == "comp":
            feature_loss_fg = F.normalize(rendered["feature_fg"], 2,1) - batch["feature"]
            feature_loss_fg = feature_loss_fg.norm(2, 1, keepdim=True)
            feature_loss_fg = feature_loss_fg * batch["mask"].float()
            loss_dict["feature"] = feature_loss_fg

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
        if self.config["use_gui"]:
            self.gui.update()
            while self.gui.pause_training:
                self.gui.update()
        self.process_frameid(batch)
        Kmat, w2c = self.compute_camera_samples(batch, self.config["train_res"])
        frameid = batch["frameid"]

        # TODO: get deformation before rendering
        self.gaussians.update_trajectory(frameid)

        # render reference view
        rendered, pts_dict = self.render_pair(self.config["train_res"], Kmat, w2c=w2c, frameid=frameid)
        self.reshape_batch(rendered)

        # prepare reference view GT
        self.reshape_batch(batch)
        self.NHWC2NCHW(batch)

        loss_dict = {}
        self.compute_recon_losses(loss_dict, rendered, batch)
        self.mask_losses(loss_dict, batch, rendered["alpha"], self.config)

        # sampled recon loss
        if self.config["xyz_wt"] > 0:
            loss_dict["xyz"],_,_ = self.gaussians.feature_matching_loss(
                                            batch["feature"], 
                                            rendered["xyz"], 
                                            batch["mask"])

        # compute regularization loss
        self.compute_reg_loss(loss_dict, frameid)

        # weight each loss term
        self.apply_loss_weights(loss_dict, self.config)

        # update per-point stats
        self.gaussians.update_vis_stats(pts_dict)
        return loss_dict

    def compute_reg_loss(self, loss_dict, frameid):
        if self.config["reg_least_deform_wt"] > 0:
            loss_dict["reg_least_deform"] = self.gaussians.get_least_deform_loss()
        if self.config["reg_least_action_wt"] > 0:
            loss_dict["reg_least_action"] = self.gaussians.get_least_action_loss()
        if self.config["reg_arap_wt"] > 0:
            loss_dict["reg_arap"] = self.gaussians.get_arap_loss(
                frameid=frameid[0, 0], frameid_2=frameid[0, 1]
            )
        if self.config["reg_lab4d_wt"] > 0:
            loss_dict["reg_lab4d"] = self.gaussians.get_lab4d_loss(frameid)

        if self.config["reg_gauss_skin_wt"] > 0:
            loss_dict["reg_gauss_skin"] = self.gaussians.gauss_skin_consistency_loss()

        if self.config["reg_skel_prior_wt"] > 0:
            loss_dict["reg_skel_prior"] = self.gaussians.skel_prior_loss()

        # TODO implement soft deform loss here

    @staticmethod
    def mask_losses(loss_dict, batch, mask_pred, config):
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
        maskfg = batch["mask"]
        vis2d = batch["vis2d"]

        # always mask-out non-visible (out-of-frame) pixels
        keys_allpix = ["mask"]
        # field type specific keys
        keys_type_specific = ["flow", "rgb", "depth", "feature"]
        # rendered-mask weighted losses
        keys_mask_weighted = ["flow", "rgb", "depth", "feature"]

        # type-specific masking rules
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
            if k in keys_allpix:
                loss_dict[k] = v * vis2d
            elif k in keys_type_specific:
                loss_dict[k] = v * mask
            else:
                raise ValueError(f"loss {k} not defined")

        # apply mask weights
        for k in keys_mask_weighted:
            loss_dict[k] *= mask_pred.detach()

        is_detected = batch["is_detected"].float()[:, None, None, None]

        # remove mask loss for frames without detection
        if config["field_type"] == "fg" or config["field_type"] == "comp":
            loss_dict["mask"] = loss_dict["mask"] * is_detected
            loss_dict["feature"] = loss_dict["feature"] * is_detected

    @staticmethod
    def apply_loss_weights(loss_dict, config):
        """Weigh each loss term according to command-line configs

        Args:
            loss_dict (Dict): Loss values for each loss term
            config (Dict): Command-line options
        """
        px_unit_keys = ["flow"]
        for k, v in loss_dict.items():
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

    def construct_rand_batch(self, num_imgs):
        """
        Returns:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16),
                "hxy" (M,N,3), and "frameid" (M,)
        """
        opts = self.config
        batch = {}
        inst_id = np.random.randint(
            0, len(self.data_info["frame_info"]["frame_offset"]) - 1, (num_imgs,)
        )
        frameid = np.random.randint(0, self.data_info["total_frames"], (num_imgs,))
        frameid[:] = 0
        frameid_sub = frameid - self.data_info["frame_info"]["frame_offset"][inst_id]

        camera_int = np.zeros((len(frameid_sub), 4))
        camera_int[:, :2] = opts["train_res"] * 2
        camera_int[:, 2:] = opts["train_res"] / 2

        from lab4d.utils.camera_utils import (
            get_object_to_camera_matrix,
            construct_batch,
        )

        field2cam = {"fg": []}
        for idx in range(len(frameid_sub)):
            theta = np.random.rand(1) * 360
            axis = np.random.rand(3)
            distance = 10
            rtmat = get_object_to_camera_matrix(theta, axis, distance)
            field2cam["fg"].append(rtmat)
        field2cam["fg"] = np.stack(field2cam["fg"], 0)

        batch = construct_batch(
            inst_id=inst_id,
            frameid_sub=frameid_sub,
            eval_res=opts["train_res"],
            field2cam=field2cam,
            camera_int=camera_int,
            crop2raw=None,
            device=self.device,
        )
        inst_id = torch.tensor(
            self.data_info["frame_info"]["frame_offset"][inst_id], device=self.device
        )
        batch["frameid"] = batch["frameid_sub"] + inst_id
        return batch


    @staticmethod
    def reshape_batch_inv(batch):
        """Reshape a batch to merge the pair dimension into the batch dimension

        Args:
            batch (Dict): Arbitrary dataloader outputs (M*2, ...). This is
                modified in place to reshape each value to (M, 2, ...)
        """
        for k, v in batch.items():
            if isinstance(v, dict):
                GSplatModel.reshape_batch_inv(v)
            else:
                batch[k] = v.reshape(-1, 2, *v.shape[1:])

    @staticmethod
    def reshape_batch(batch):
        """Reshape a batch to merge the pair dimension into the batch dimension

        Args:
            batch (Dict): Arbitrary dataloader outputs (M, 2, ...). This is
                modified in place to reshape each value to (M*2, ...)
        """
        for k, v in batch.items():
            batch[k] = v.view(-1, *v.shape[2:])

    @staticmethod
    def NCHW2NHWC(batch):
        """Convert batch from NCHW to NHWC

        Args:
            batch (Dict): Arbitrary dataloader outputs (M, ...). This is
                modified in place to convert each value from NCHW to NHWC
        """
        for k, v in batch.items():
            if v.ndim == 4:
                batch[k] = v.permute(0, 2, 3, 1)

    @staticmethod
    def NHWC2NCHW(batch):
        """
        Args:
            batch (Dict): Arbitrary dataloader outputs (M, ...). This is
                modified in place to convert each value from NHWC to NCHW
        """
        for k, v in batch.items():
            if v.ndim == 4:
                batch[k] = v.permute(0, 3, 1, 2)

    def augment_visualization_nv(self, rendered, cam_dict, Kmat, w2c, frameid):
        # modify w2c to be a turn-table view
        num_nv = 8
        w2c_nv = get_rotating_cam(num_nv, max_angle=360)
        w2c_nv = torch.tensor(w2c_nv, dtype=torch.float32, device=w2c.device)
        w2c_nv[..., :3, 3] = w2c[0, 0, :3, 3]
        w2c_nv = w2c_nv.reshape(-1, 2, 4, 4)
        Kmat_nv = Kmat[:1, :1].repeat(num_nv // 2, 2, 1, 1)
        frameid_nv = frameid[:1, :1].repeat(num_nv // 2, 2)
        rendered_nv,_ = self.render_pair(
            cam_dict, Kmat_nv, w2c=w2c_nv, frameid=frameid_nv
        )
        for k, v in rendered_nv.items():
            rendered[k] = torch.cat([rendered[k], v], 0)

    @torch.no_grad()
    def evaluate(self, batch, is_pair=True, augment_nv=True, render_flow=True, return_numpy=True):
        """Evaluate model on a batch of data"""
        self.process_frameid(batch)
        if not is_pair:
            # fake a pair
            for k, v in batch.items():
                batch[k] = fake_a_pair(v)
        else:
            self.reshape_batch_inv(batch)
        # render mode or eval mode during training
        if "render_res" in self.config.keys():
            crop_size = self.config["render_res"]
        else:
            crop_size = self.config["eval_res"]
        Kmat, w2c = self.compute_camera_samples(batch, crop_size)
        frameid = batch["frameid"]

        # TODO: get deformation before rendering
        self.gaussians.update_trajectory(frameid)

        rendered, _ = self.render_pair(crop_size, Kmat, w2c=w2c, frameid=frameid, device="cpu")
        for k, v in rendered.items():
            rendered[k] = v[:, 0]

        scalars = {}
        out_dict = {"rgb": [], "depth": [], "alpha": [], "xyz": [], "flow": [], "mask_fg": [], "feature": [], "vis2d": []}
        for k, v in rendered.items():
            if k in out_dict.keys():
                v = v.permute(0, 2, 3, 1)
                if return_numpy:
                    v = v.cpu().numpy()
                out_dict[k] = v

        return out_dict, scalars

    def process_frameid(self, batch):
        """Convert frameid within each video to overall frame id

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16), and
                "hxy" (M,N,3). This function modifies it in place to add key
                "frameid" (M,)
        """
        # convert frameid_sub to frameid
        if "motion_id" in batch.keys():
            # indicator for reanimation
            motion_id = batch["motion_id"]
            del batch["motion_id"]
        else:
            motion_id = batch["dataid"]
        frame_offset_raw = torch.tensor(
            self.data_info["frame_info"]["frame_offset_raw"], device=batch["frameid_sub"].device
        )
        batch["frameid"] = batch["frameid_sub"] + frame_offset_raw[motion_id]

    def set_progress(self, current_steps, progress, sub_progress):
        """Adjust loss weights and other constants throughout training

        Args:
            current_steps (int): Number of optimization steps so far
            progress (float): Fraction of training completed (in the current stage)
        """
        self.progress = progress
        self.current_steps = current_steps
        config = self.config

        # local vs global arap loss
        if self.progress > config["inc_warmup_ratio"]:
            self.gaussians.is_inc_mode = False
        else:
            self.gaussians.is_inc_mode = True

        # knn for arap
        anchor_x = (0, 1.0)
        anchor_y = (1.0, 0.0)
        type = "linear"
        if self.progress > config["inc_warmup_ratio"]:
            ratio_knn = interp_wt(anchor_x, anchor_y, progress, type=type)
        else:
            ratio_knn = interp_wt(anchor_x, anchor_y, sub_progress, type=type)
        self.gaussians.ratio_knn = ratio_knn

        # arap wt
        loss_name = "reg_arap_wt"
        anchor_x = (0, 200.0)
        anchor_y = (0.0, 1.0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # least deform wt 
        loss_name = "reg_least_deform_wt"
        anchor_x = (0, 1.0)
        anchor_y = (1.0, 0.0)
        type = "linear"
        if self.progress > config["inc_warmup_ratio"]:
            self.set_loss_weight(loss_name, anchor_x, anchor_y, progress, type=type)
        else:
            self.set_loss_weight(loss_name, anchor_x, anchor_y, sub_progress, type=type)

        # least action wt
        loss_name = "reg_least_action_wt"
        anchor_x = (0, 1.0)
        anchor_y = (1.0, 0.0)
        type = "linear"
        if self.progress > config["inc_warmup_ratio"]:
            self.set_loss_weight(loss_name, anchor_x, anchor_y, progress, type=type)
        else:
            self.set_loss_weight(loss_name, anchor_x, anchor_y, sub_progress, type=type)

    def set_loss_weight(
        self, loss_name, anchor_x, anchor_y, progress_ratio, type="linear"
    ):
        """Set a loss weight according to the current training step

        Args:
            loss_name (str): Name of loss weight to set
            anchor_x: Tuple of optimization steps [x0, x1]
            anchor_y: Tuple of loss values [y0, y1]
            progress_ratio (float): Current optimization ratio, 0 to 1
            type (str): Interpolation type ("linear" or "log")
        """
        if f"{loss_name}_init" not in self.config.keys():
            self.config[f"{loss_name}_init"] = self.config[loss_name]
        factor = interp_wt(anchor_x, anchor_y, progress_ratio, type=type)
        self.config[loss_name] = self.config[f"{loss_name}_init"] * factor

    @torch.no_grad()
    def get_field_params(self):
        """Get beta values for all neural fields

        Returns:
            betas (Dict): Beta values for each neural field
        """
        beta_dicts = {"num_pts": self.gaussians.get_num_pts}
        return beta_dicts

    def convert_img_to_pixel(self, batch):
        """Check that batch shape matches pixel array, otherwise convert to expected shape
        The same as dvr_renderer.convert_img_to_pixel

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2),
                "hxy" (M,2,N,3), and "is_detected" (M,2)
        """
        for k, v in batch.items():
            if len(v.shape) == 5:
                M, _, H, W, K = v.shape
                batch[k] = v.view(M, -1, H * W, K)

    def update_geometry_aux(self):
        """Extract proxy geometry for all neural fields"""
        self.gaussians.update_geometry_aux()

    def update_camera_aux(self):
        pass

    def export_geometry_aux(self, path):
        """Export proxy geometry for all neural fields"""
        self.gaussians.export_geometry_aux(path)

    @torch.no_grad()
    def get_cameras(self, frame_id=None):
        """Compute camera matrices in world units

        Returns:
            field2cam (Dict): Maps field names ("fg" or "bg") to (M,4,4) cameras
        """
        field2cam = {}
        field2cam["fg"] = self.gaussians.get_extrinsics(frameid=frame_id)
        return field2cam

    @torch.no_grad()
    def get_intrinsics(self, frame_id=None):
        """Compute camera intrinsics at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute at all frames
        Returns:
            intrinsics: (..., 4) Output camera intrinsics
        """
        return self.intrinsics.get_vals(frame_id=frame_id)

    @torch.no_grad()
    def get_aabb(self, inst_id=None):
        """Compute axis aligned bounding box
        Args:
            inst_id (int or tensor): Instance id. If None, return aabb for all instances

        Returns:
            aabb (Dict): Maps field names ("fg" or "bg") to (1/N,2,3) aabb
        """
        aabb = {}
        aabb["fg"] = self.gaussians.get_aabb()[None]
        return aabb

    def mlp_init(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics for all neural fields from external priors
        """
        self.gaussians.init_camera_mlp()
        self.gaussians.init_deform_mlp()


if __name__ == "__main__":
    import cv2
    import os, sys

    from lab4d.utils.io import save_vid

    opts = {"sh_degree": 0}
    renderer = GSplatModel(opts, None)

    # convert this to a dict
    K = np.array([2, 2, 0, 0])
    Kmat = K2mat(K)
    cam_dict = renderer.get_default_cam(render_resolution=512, Kmat=Kmat)
    res = cam_dict["render_resolution"]
    w2c = np.eye(4)
    w2c[2, 3] = 3  # depth

    # render turntable view
    nframes = 10
    frames = []
    for i in range(nframes):
        w2c[:3, :3] = cv2.Rodrigues(
            np.asarray([0.0, 2 * np.pi * (0.25 + i / nframes), 0.0])
        )[0]
        out,_ = renderer.render(cam_dict, w2c=w2c)
        img = out["rgb"][0].permute(1, 2, 0).detach().cpu().numpy()
        frames.append(img)
        cv2.imwrite(f"tmp/{i:02d}.png", img * 255)
    save_vid("tmp/vid", frames)
    print("saved to tmp/vid.mp4")
