# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import trimesh
from torch import nn
from torch.nn import functional as F
import sys
import os

os.environ["CUDA_PATH"] = sys.prefix  # needed for geomloss
from geomloss import SamplesLoss

from lab4d.nnutils.feature import FeatureNeRF
from lab4d.nnutils.warping import SkinningWarp, create_warp
from lab4d.utils.decorator import train_only_fields
from lab4d.utils.geom_utils import extend_aabb, check_inside_aabb


class Deformable(FeatureNeRF):
    """A dynamic neural radiance field

    Args:
        fg_motion (str): Foreground motion type ("rigid", "dense", "bob",
            "skel-{human,quad}", or "comp_skel-{human,quad}_{bob,dense}")
        data_info (Dict): Dataset metadata from get_data_info()
        D (int): Number of linear layers for density (sigma) encoder
        W (int): Number of hidden units in each MLP layer
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_dir (int): Number of frequencies in direction embedding
        appr_channels (int): Number of channels in the global appearance code
            (captures shadows, lighting, and other environmental effects)
        appr_num_freq_t (int): Number of frequencies in the time embedding of
            the global appearance code
        num_inst (int): Number of distinct object instances. If --nosingle_inst
            is passed, this is equal to the number of videos, as we assume each
            video captures a different instance. Otherwise, we assume all videos
            capture the same instance and set this to 1.
        inst_channels (int): Number of channels in the instance code
        skips (List(int): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        init_beta (float): Initial value of beta, from Eqn. 3 of VolSDF.
            We transform a learnable signed distance function into density using
            the CDF of the Laplace distribution with zero mean and beta scale.
        init_scale (float): Initial geometry scale factor.
        color_act (bool): If True, apply sigmoid to the output RGB
        feature_channels (int): Number of feature field channels
    """

    def __init__(
        self,
        fg_motion,
        data_info,
        D=8,
        W=256,
        num_freq_xyz=10,
        num_freq_dir=4,
        appr_channels=32,
        appr_num_freq_t=6,
        num_inst=1,
        inst_channels=32,
        skips=[4],
        activation=nn.ReLU(True),
        init_beta=0.1,
        init_scale=0.1,
        color_act=True,
        feature_channels=16,
    ):
        super().__init__(
            data_info,
            D=D,
            W=W,
            num_freq_xyz=num_freq_xyz,
            num_freq_dir=num_freq_dir,
            appr_channels=appr_channels,
            appr_num_freq_t=appr_num_freq_t,
            num_inst=num_inst,
            inst_channels=inst_channels,
            skips=skips,
            activation=activation,
            init_beta=init_beta,
            init_scale=init_scale,
            color_act=color_act,
            feature_channels=feature_channels,
        )

        self.warp = create_warp(fg_motion, data_info)
        self.fg_motion = fg_motion

    def init_proxy(self, geom_path, init_scale):
        """Initialize proxy geometry as a sphere

        Args:
            geom_path (str): Unused
            init_scale (float): Unused
        """
        self.proxy_geometry = trimesh.creation.uv_sphere(radius=0.12, count=[4, 4])

    def get_init_sdf_fn(self):
        """Initialize signed distance function as a skeleton or sphere

        Returns:
            sdf_fn_torch (Function): Signed distance function
        """

        def sdf_fn_torch_sphere(pts):
            radius = 0.1
            # l2 distance to a unit sphere
            dis = (pts).pow(2).sum(-1, keepdim=True)
            sdf = torch.sqrt(dis) - radius  # negative inside, postive outside
            return sdf

        @torch.no_grad()
        def sdf_fn_torch_skel(pts):
            sdf = self.warp.get_gauss_sdf(pts)
            return sdf

        if "skel-" in self.fg_motion or "urdf-" in self.fg_motion:
            return sdf_fn_torch_skel
        else:
            return sdf_fn_torch_sphere

    def backward_warp(
        self, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict={}
    ):
        """Warp points from camera space to object canonical space. This
        requires "un-articulating" the object from observed time-t to rest.

        Args:
            xyz_cam: (M,N,D,3) Points along rays in camera space
            dir_cam: (M,N,D,3) Ray directions in camera space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance.
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            xyz: (M,N,D,3) Points along rays in object canonical space
            dir: (M,N,D,3) Ray directions in object canonical space
            xyz_t: (M,N,D,3) Points along rays in object time-t space.
        """
        xyz_t, dir = self.cam_to_field(xyz_cam, dir_cam, field2cam)
        xyz, warp_dict = self.warp(
            xyz_t, frame_id, inst_id, type="backward", samples_dict=samples_dict, return_aux=True
        )

        # TODO: apply se3 to dir
        backwarp_dict = {"xyz": xyz, "dir": dir, "xyz_t": xyz_t}
        backwarp_dict.update(warp_dict)
        return backwarp_dict

    def forward_warp(self, xyz, field2cam, frame_id, inst_id, samples_dict={}):
        """Warp points from object canonical space to camera space. This
        requires "re-articulating" the object from rest to observed time-t.

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            xyz_cam: (M,N,D,3) Points along rays in camera space
        """
        xyz_next = self.warp(
            xyz, frame_id, inst_id, type="forward", samples_dict=samples_dict
        )
        xyz_cam = self.field_to_cam(xyz_next, field2cam)
        return xyz_cam

    def flow_warp(
        self,
        xyz_1,
        field2cam_flip,
        frame_id,
        inst_id,
        samples_dict={},
    ):
        """Warp points from camera space from time t1 to time t2

        Args:
            xyz_1: (M,N,D,3) Points along rays in canonical space at time t1
            field2cam_flip: (M,SE(3)) Object-to-camera SE(3) transform at time t2
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))

        Returns:
            xyz_2: (M,N,D,3) Points along rays in camera space at time t2
        """
        xyz_2 = self.warp(
            xyz_1, frame_id, inst_id, type="flow", samples_dict=samples_dict
        )
        xyz_2 = self.field_to_cam(xyz_2, field2cam_flip)
        return xyz_2

    @train_only_fields
    def cycle_loss(self, xyz, xyz_t, frame_id, inst_id, samples_dict={}):
        """Enforce cycle consistency between points in object canonical space,
        and points warped from canonical space, backward to time-t space, then
        forward to canonical space again

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            xyz_t: (M,N,D,3) Points along rays in object time-t space
            frame_id: (M,) Frame id. If None, render at all frames
            inst_id: (M,) Instance id. If None, render for the average instance
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            cyc_dict (Dict): Cycle consistency loss. Keys: "cyc_dist" (M,N,D,1)
        """
        cyc_dict = super().cycle_loss(xyz, xyz_t, frame_id, inst_id, samples_dict)

        xyz_t_cycled, warp_dict = self.warp(
            xyz, frame_id, inst_id, type="forward", samples_dict=samples_dict, return_aux=True
        )
        cyc_dist = (xyz_t_cycled - xyz_t).norm(2, dim=-1, keepdim=True)
        cyc_dict["cyc_dist"] = cyc_dist
        cyc_dict.update(warp_dict)
        return cyc_dict

    def gauss_skin_consistency_loss(self, type="optimal_transport"):
        """Enforce consistency between the NeRF's SDF and the SDF of Gaussian bones,

        Args:
            type (str): "optimal_transport" or "density"
        Returns:
            loss: (0,) Skinning consistency loss
        """
        if type == "optimal_transport":
            return self.gauss_optimal_transport_loss()
        elif type == "density":
            return self.gauss_skin_density_loss()
        else:
            raise NotImplementedError

    def gauss_skin_density_loss(self, nsample=4096):
        """Enforce consistency between the NeRF's SDF and the SDF of Gaussian bones,
        based on density.

        Args:
            nsample (int): Number of samples to take from both distance fields
        Returns:
            loss: (0,) Skinning consistency loss
        """
        pts, frame_id, _ = self.sample_points_aabb(nsample, extend_factor=0.5)
        inst_id = None
        samples_dict = {}
        t_articulation, rest_articulation = self.warp.articulation.get_vals_and_mean(frame_id)
        samples_dict["t_articulation"] = t_articulation
        samples_dict["rest_articulation"] = rest_articulation

        # match the gauss density to the reconstructed density
        bones2obj = samples_dict["t_articulation"]
        bones2obj = (
            torch.cat([bones2obj[0], samples_dict["rest_articulation"][0]], dim=0),
            torch.cat([bones2obj[1], samples_dict["rest_articulation"][1]], dim=0),
        )
        pts_gauss = torch.cat([pts, pts], dim=0)
        density_gauss = self.warp.get_gauss_density(pts_gauss, bone2obj=bones2obj)

        # match gauss density to reconstructed density for clothing layer
        if hasattr(self.warp, "post_warp") and isinstance(self.warp.post_warp, SkinningWarp):
            t_articulation_cloth, rest_articulation_cloth = (
                self.warp.post_warp.articulation.get_vals_and_mean(frame_id)
            )

            bones2obj_cloth = t_articulation
            bones2obj_cloth = (
                torch.cat([bones2obj_cloth[0], rest_articulation_cloth[0]], dim=0),
                torch.cat([bones2obj_cloth[1], rest_articulation_cloth[1]], dim=0),
            )
            pts_gauss_cloth = torch.cat([pts, pts], dim=0)
            density_gauss_cloth = self.warp.post_warp.get_gauss_density(pts_gauss_cloth, bone2obj=bones2obj_cloth)
            density_gauss = density_gauss + density_gauss_cloth

        with torch.no_grad():
            density = torch.zeros_like(density_gauss)
            pts_warped = self.warp(
                pts[:, None, None],
                frame_id,
                inst_id,
                type="backward",
                samples_dict=samples_dict,
                return_aux=False,
            )[:, 0, 0]
            pts = torch.cat([pts_warped, pts], dim=0)

            # check whether the point is inside the aabb
            aabb = self.get_aabb()
            aabb = extend_aabb(aabb)
            inside_aabb = check_inside_aabb(pts, aabb)

            _, density[inside_aabb] = self.forward(pts[inside_aabb], inst_id=inst_id)
            density = density / self.logibeta.exp()  # (0,1)

        # loss = ((density_gauss - density).pow(2)).mean()
        # binary cross entropy loss to align gauss density to the reconstructed density
        # weight the loss such that:
        # wp lp = wn ln
        # wp lp + wn ln = lp + ln
        weight_pos = 0.5 / (1e-6 + density.mean())
        weight_neg = 0.5 / (1e-6 + 1 - density).mean()
        weight = density * weight_pos + (1 - density) * weight_neg
        loss = ((density_gauss - density).pow(2) * weight.detach()).mean()
        return loss

    def gauss_optimal_transport_loss(self, nsample=1024):
        """Enforce consistency between the NeRF's proxy rest shape
         and the gaussian bones, based on optimal transport.

        Args:
            nsample (int): Number of samples to take from proxy geometry
        Returns:
            loss: (0,) Gaussian optimal transport loss
        """
        # optimal transport loss
        device = self.parameters().__next__().device
        pts_proxy = self.get_proxy_geometry().vertices  # N_pts, 3

        samploss = SamplesLoss(loss="sinkhorn", p=2, blur=0.002, scaling=0.5, truncate=1)
        scale_proxy = self.get_scale()  # to normalize pts to 1

        # sample points from the proxy geometry
        pts_proxy = pts_proxy[np.random.choice(len(pts_proxy), nsample)]  # N_sample, 3
        pts_proxy = torch.tensor(pts_proxy, dtype=torch.float32, device=device)  # N_sample, 3

        if hasattr(self.warp, "post_warp") and isinstance(self.warp.post_warp, SkinningWarp):
            pts_gauss_body = self.warp.get_gauss_pts()  # N_body, 3
            pts_gauss_cloth = self.warp.post_warp.get_gauss_pts()  # N_cloth, 3
            pts_gauss_both = torch.cat([pts_gauss_body, pts_gauss_cloth], dim=-2)  # N_body+N_cloth, 3

            # Apply two Sinkhorn loss: (A) sampling from both body and clothing gaussians,
            # (B) sampling from clothing gaussians only.
            # Intent of (A) is to keep body gaussians within a subset of the SDF shape.
            # Intent of (B) is to encourage clothing gaussians to cover the entire SDF shape.
            loss = (
                samploss(2 * pts_gauss_both / scale_proxy, 2 * pts_proxy / scale_proxy).mean() +
                samploss(2 * pts_gauss_cloth / scale_proxy, 2 * pts_proxy / scale_proxy).mean()
            ) / 2.
        else:
            # Apply Sinkhorn loss: sample points from body gaussians
            # Intent is to encourage distribution of body gaussians to match SDF shape
            pts_gauss_body = self.warp.get_gauss_pts()  # N_body, 3
            loss = samploss(2 * pts_gauss_body / scale_proxy, 2 * pts_proxy / scale_proxy).mean()

        return loss

    def soft_deform_loss(self, nsample=1024):
        """Minimize soft deformation so it doesn't overpower the skeleton.
        Compute L2 distance of points before and after soft deformation

        Args:
            nsample (int): Number of samples to take from both distance fields
        Returns:
            loss: (0,) Soft deformation loss
        """
        pts, frame_id, inst_id = self.sample_points_aabb(nsample, extend_factor=1.0)
        dist2 = self.warp.compute_post_warp_dist2(pts[:, None, None], frame_id, inst_id)
        return dist2.mean()

    def get_samples(self, Kinv, batch):
        """Compute time-dependent camera and articulation parameters.

        Args:
            Kinv: (N,3,3) Inverse of camera matrix
            Batch (Dict): Batch of inputs. Keys: "dataid", "frameid_sub",
                "crop2raw", "feature", "hxy", and "frameid"
        Returns:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2),
                "feature" (M,N,16), "rest_articulation" ((M,B,4), (M,B,4)), and
                "t_articulation" ((M,B,4), (M,B,4))
        """
        samples_dict = super().get_samples(Kinv, batch)

        if isinstance(self.warp, SkinningWarp):
            # cache the articulation values
            # mainly to avoid multiple fk computation
            # (M,K,4)x2, # (M,K,4)x2
            inst_id = samples_dict["inst_id"]
            frame_id = samples_dict["frame_id"]
            if "joint_so3" in batch.keys():
                override_so3 = batch["joint_so3"]
                samples_dict[
                    "rest_articulation"
                ] = self.warp.articulation.get_mean_vals()
                samples_dict["t_articulation"] = self.warp.articulation.get_vals(
                    frame_id, override_so3=override_so3
                )
            else:
                (
                    samples_dict["t_articulation"],
                    samples_dict["rest_articulation"],
                ) = self.warp.articulation.get_vals_and_mean(frame_id)
        return samples_dict

    def mlp_init(self):
        """For skeleton fields, initialize bone lengths and rest joint angles
        from an external skeleton
        """
        if "skel-" in self.fg_motion or "urdf-" in self.fg_motion:
            if hasattr(self.warp.articulation, "init_vals"):
                self.warp.articulation.mlp_init()
        # Note: mlp_init() initializes the SDF shape to the articulated gaussian bones,
        # so should be called after the warp articulation is initialized
        super().mlp_init()

    def query_field(self, samples_dict, flow_thresh=None):
        """Render outputs from a neural radiance field.

        Args:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2), and
                "feature" (M,N,16), "rest_articulation" ((M,B,4), (M,B,4)),
                and "t_articulation" ((M,B,4), (M,B,4))
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            feat_dict (Dict): Neural field outputs. Keys: "rgb" (M,N,D,3),
                "density" (M,N,D,1), "density_{fg,bg}" (M,N,D,1), "vis" (M,N,D,1),
                "cyc_dist" (M,N,D,1), "xyz" (M,N,D,3), "xyz_cam" (M,N,D,3),
                "depth" (M,1,D,1) TODO
            deltas: (M,N,D,1) Distance along rays between adjacent samples
            aux_dict (Dict): Auxiliary neural field outputs. Keys: TODO
        """
        feat_dict, deltas, aux_dict = super().query_field(
            samples_dict, flow_thresh=flow_thresh
        )

        # xyz = feat_dict["xyz"].detach()  # don't backprop to cam/dfm fields
        xyz = feat_dict["xyz"]
        xyz_t = feat_dict["xyz_t"]
        gauss_field = self.compute_gauss_density(xyz, xyz_t, samples_dict)
        feat_dict.update(gauss_field)

        return feat_dict, deltas, aux_dict

    def compute_gauss_density(self, xyz, xyz_t, samples_dict):
        """If this is a SkinningWarp, compute density from Gaussian bones

        Args:
            xyz: (M,N,D,3) Points in object canonical space
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2), and
                "feature" (M,N,16), "rest_articulation" ((M,B,4), (M,B,4)),
                and "t_articulation" ((M,B,4), (M,B,4))
        Returns:
            gauss_field (Dict): Density. Keys: "gauss_density" (M,N,D,1)
        """
        M, N, D, _ = xyz.shape
        gauss_field = {}
        if isinstance(self.warp, SkinningWarp):
            # supervise t articulation
            xyz_t = xyz_t.view(-1, 3).detach()
            t_articulation = (
                samples_dict["t_articulation"][0][:, None]
                .repeat(1, N * D, 1, 1)
                .view(M * N * D, -1, 4),
                samples_dict["t_articulation"][1][:, None]
                .repeat(1, N * D, 1, 1)
                .view(M * N * D, -1, 4),
            )
            gauss_density = self.warp.get_gauss_density(xyz_t, bone2obj=t_articulation)

            # supervise rest articulation
            gauss_density = gauss_density * self.warp.logibeta.exp()
            gauss_field["gauss_density"] = gauss_density.view((M, N, D, 1))

        return gauss_field
