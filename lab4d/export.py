# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
"""
python lab4d/export.py --flagfile=logdir/cat-85-sub-sub-bob-pika-cate-b02/opts.log --load_suffix latest --inst_id 0
"""

import os, sys
import json
from typing import NamedTuple, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from absl import app, flags
import tqdm
import signal

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.config import get_config
from lab4d.dataloader import data_utils
from lab4d.engine.trainer import Trainer
from lab4d.nnutils.warping import SkinningWarp
from lab4d.nnutils.pose import ArticulationSkelMLP
from lab4d.utils.io import make_save_dir, save_rendered
from lab4d.utils.quat_transform import (
    dual_quaternion_to_se3,
    quaternion_translation_to_se3,
)
from lab4d.utils.vis_utils import append_xz_plane

cudnn.benchmark = True


class ExportMeshFlags:
    flags.DEFINE_integer("inst_id", 0, "video/instance id")
    flags.DEFINE_integer("grid_size", 256, "grid size of marching cubes")
    flags.DEFINE_float(
        "level", 0.0015, "contour value of marching cubes use to search for isosurfaces"
    )
    flags.DEFINE_float(
        "vis_thresh", 0.0, "visibility threshold to remove invisible pts, -inf to inf"
    )
    flags.DEFINE_boolean("extend_aabb", False, "use extended aabb for meshing (for bg)")
    flags.DEFINE_string("flag", "", "optional flags: '' (default), 'body_only', 'cloth_only', 'canonical', 'bone_len_change'")


class MotionParamsExpl(NamedTuple):
    """
    explicit motion params for reanimation and transfer
    """

    field2cam: Tuple[torch.Tensor, torch.Tensor]  # (quaternion, translation)
    t_articulation: Tuple[
        torch.Tensor, torch.Tensor
    ]  # dual quaternion, applies to skinning
    so3: torch.Tensor  # so3, applies to skeleton
    mesh_t: trimesh.Trimesh  # mesh at time t
    bone_t: trimesh.Trimesh  # bone center at time t


def extract_deformation(field, mesh_rest, inst_id, flag=""):
    # get corresponding frame ids
    frame_mapping = field.camera_mlp.frame_mapping
    frame_offset = field.frame_offset
    frame_ids = frame_mapping[frame_offset[inst_id] : frame_offset[inst_id + 1]]
    start_id = frame_ids[0]
    print("Extracting motion parameters for inst id:", inst_id)
    # print("Frame ids with the video:", frame_ids - start_id)

    device = next(field.parameters()).device
    xyz = torch.tensor(mesh_rest.vertices, dtype=torch.float32, device=device)
    inst_id = torch.tensor([inst_id], dtype=torch.long, device=device)

    motion_tuples = {}
    for frame_id in frame_ids:
        frame_id = frame_id[None]
        field2cam = field.camera_mlp.get_vals(frame_id)

        samples_dict = {}
        se3_mat = quaternion_translation_to_se3(field2cam[0], field2cam[1])[0]
        se3_mat = se3_mat.cpu().numpy()
        if hasattr(field, "warp") and isinstance(field.warp, SkinningWarp):
            if flag == "canonical":
                # Canonical shape: Disable all deformation
                t_articulation = None
                so3 = None
                mesh_bones_t = None

            elif flag == "cloth_only":
                # Show clothing deformation only: Only render gaussians from post_warp
                if hasattr(field.warp, "post_warp") and isinstance(field.warp.post_warp, SkinningWarp):
                    t_articulation, rest_articulation = (
                        field.warp.post_warp.articulation.get_vals_and_mean(frame_id)
                    )
                    samples_dict["t_articulation"] = t_articulation
                    samples_dict["rest_articulation"] = rest_articulation

                    if isinstance(field.warp.post_warp.articulation, ArticulationSkelMLP):
                        so3 = field.warp.post_warp.articulation.get_vals(frame_id, return_so3=True)[0].cpu().numpy()
                    else:
                        so3 = None

                    mesh_bones_t = field.warp.post_warp.skinning_model.draw_gaussian(
                        (t_articulation[0][0], t_articulation[1][0]),
                        field.warp.post_warp.articulation.edges,
                    )

                    # Make clothing gaussians appear red-yellow
                    mesh_bones_t.visual.vertex_colors[:, 2] = 0
                    mesh_bones_t.visual.vertex_colors[:, 0] = (
                        128 + mesh_bones_t.visual.vertex_colors[:, 0] // 2
                    )

                else:
                    t_articulation = None
                    so3 = None
                    mesh_bones_t = None

            elif flag == "body_only":
                # Show body deformation only: Only render gaussians from base articulation
                t_articulation, rest_articulation = field.warp.articulation.get_vals_and_mean(frame_id)
                samples_dict["t_articulation"] = t_articulation
                samples_dict["rest_articulation"] = rest_articulation

                if isinstance(field.warp.articulation, ArticulationSkelMLP):
                    so3 = field.warp.articulation.get_vals(frame_id, return_so3=True)[0].cpu().numpy()
                else:
                    so3 = None

                mesh_bones_t = field.warp.skinning_model.draw_gaussian(
                    (t_articulation[0][0], t_articulation[1][0]),
                    field.warp.articulation.edges,
                )

                # Make body gaussians appear blue-cyan
                mesh_bones_t.visual.vertex_colors[:, 0] = 0
                mesh_bones_t.visual.vertex_colors[:, 2] = (
                    128 + mesh_bones_t.visual.vertex_colors[:, 2] // 2
                )

            else:
                # Show both body deformation, and clothing deformation warped by body, if they exist
                t_articulation, rest_articulation = field.warp.articulation.get_vals_and_mean(frame_id)
                samples_dict["t_articulation"] = t_articulation
                samples_dict["rest_articulation"] = rest_articulation

                if isinstance(field.warp.articulation, ArticulationSkelMLP):
                    so3 = field.warp.articulation.get_vals(frame_id, return_so3=True)[0].cpu().numpy()
                else:
                    so3 = None

                mesh_bones_t = field.warp.skinning_model.draw_gaussian(
                    (t_articulation[0][0], t_articulation[1][0]),
                    field.warp.articulation.edges,
                )

                if hasattr(field.warp, "post_warp") and isinstance(field.warp.post_warp, SkinningWarp):
                    t_articulation_cloth, rest_articulation_cloth = (
                        field.warp.post_warp.articulation.get_vals_and_mean(frame_id)
                    )

                    mesh_bones_t_cloth = field.warp.post_warp.skinning_model.draw_gaussian(
                        (t_articulation_cloth[0][0], t_articulation_cloth[1][0]),
                        field.warp.post_warp.articulation.edges,
                    )

                    # Forward warp clothing gaussians using body deformation model
                    xyz_bones_t_cloth = torch.tensor(
                        mesh_bones_t_cloth.vertices, dtype=torch.float32, device=device
                    )
                    xyz_bones_t_cloth = field.warp(
                        xyz_bones_t_cloth[None, None], frame_id, inst_id,
                        samples_dict=samples_dict, disable_post_warp=True,
                    )[0, 0]
                    mesh_bones_t_cloth.vertices = xyz_bones_t_cloth.detach().cpu().numpy()

                    # Make body gaussians appear blue-cyan
                    mesh_bones_t.visual.vertex_colors[:, 0] = 0
                    mesh_bones_t.visual.vertex_colors[:, 2] = (
                        128 + mesh_bones_t.visual.vertex_colors[:, 2] // 2
                    )

                    # Make clothing gaussians appear red-yellow
                    mesh_bones_t_cloth.visual.vertex_colors[:, 2] = 0
                    mesh_bones_t_cloth.visual.vertex_colors[:, 0] = (
                        128 + mesh_bones_t_cloth.visual.vertex_colors[:, 0] // 2
                    )

                    mesh_bones_t = trimesh.util.concatenate([mesh_bones_t, mesh_bones_t_cloth])

            if t_articulation is not None:
                t_articulation = dual_quaternion_to_se3(t_articulation)[0].cpu().numpy()  # 1,K,4,4
        else:
            t_articulation = None
            so3 = None
            mesh_bones_t = None

        if hasattr(field, "warp"):
            # warp mesh
            if flag == "canonical":
                xyz_t = xyz

            elif flag == "body_only":
                # Show body deformation only by disabling the post_warp.
                xyz_t = field.warp(
                    xyz[None, None], frame_id, inst_id, samples_dict=samples_dict, disable_post_warp=True,
                )[0, 0]

            elif flag == "cloth_only":
                # Show clothing deformation only by just calling the post_warp.
                # samples_dict is used to cache the body warp's forward kinematics, so
                # don't pass it to the post_warp.
                if hasattr(field.warp, "post_warp"):
                    xyz_t = field.warp.post_warp(xyz[None, None], frame_id, inst_id)[0, 0]
                else:
                    xyz_t = xyz

            else:
                # If two-layer deformation is used, show both body-layer and
                # clothing-layer deformation, as usual
                xyz_t = field.warp(
                    xyz[None, None], frame_id, inst_id, samples_dict=samples_dict
                )[0, 0]

            mesh_t = trimesh.Trimesh(
                vertices=xyz_t.cpu().numpy(), faces=mesh_rest.faces, process=False
            )
        else:
            mesh_t = mesh_rest.copy()

        motion_expl = MotionParamsExpl(
            field2cam=se3_mat,
            t_articulation=t_articulation,
            so3=so3,
            mesh_t=mesh_t,
            bone_t=mesh_bones_t,
        )
        frame_id_sub = (frame_id[0] - start_id).cpu()
        motion_tuples[frame_id_sub] = motion_expl

    # This part is for RAC project to visualize bone length change between instances
    # Can be removed in single-instance case
    if flag == "bone_len_change" and hasattr(field, "warp") and isinstance(field.warp, SkinningWarp):
        # modify rest mesh based on instance morphological changes on bones
        # idendity transformation of cameras
        field2cam_rot_idn = torch.zeros_like(field2cam[0])
        field2cam_rot_idn[..., 0] = 1.0
        field2cam_idn = (field2cam_rot_idn, torch.zeros_like(field2cam[1]))

        # bone stretching from rest to instance id
        samples_dict["t_articulation"] = field.warp.articulation.get_mean_vals(
            inst_id=inst_id
        )

        xyz_i = field.forward_warp(
            xyz[None, None],
            field2cam_idn,
            None,
            inst_id,
            samples_dict=samples_dict,
        )
        xyz_i = xyz_i[0, 0]

        xyz_i = xyz
        mesh_rest = trimesh.Trimesh(vertices=xyz_i.cpu().numpy(), faces=mesh_rest.faces)

    return mesh_rest, motion_tuples


def rescale_motion_tuples(motion_tuples, field_scale):
    """
    rescale motion tuples to world scale
    """
    for frame_id, motion_tuple in motion_tuples.items():
        motion_tuple.field2cam[:3, 3] /= field_scale
        motion_tuple.mesh_t.apply_scale(1.0 / field_scale)
        if motion_tuple.bone_t is not None:
            motion_tuple.bone_t.apply_scale(1.0 / field_scale)
        if motion_tuple.t_articulation is not None:
            motion_tuple.t_articulation[1][:] /= field_scale
    return


def save_motion_params(meshes_rest, motion_tuples, save_dir):
    for cate, mesh_rest in meshes_rest.items():
        mesh_rest.export(f"{save_dir}/{cate}-mesh.ply")
        motion_params = {"field2cam": [], "t_articulation": [], "joint_so3": []}
        os.makedirs(f"{save_dir}/fg/mesh", exist_ok=True)
        os.makedirs(f"{save_dir}/bg/mesh", exist_ok=True)
        os.makedirs(f"{save_dir}/fg/bone", exist_ok=True)
        for frame_id, motion_expl in tqdm.tqdm(motion_tuples[cate].items(), desc=f"Extracting motion params"):
            # save mesh
            motion_expl.mesh_t.export(f"{save_dir}/{cate}/mesh/{frame_id:05d}.ply")
            if motion_expl.bone_t is not None:
                motion_expl.bone_t.export(f"{save_dir}/{cate}/bone/{frame_id:05d}.ply")

            # save motion params
            motion_params["field2cam"].append(motion_expl.field2cam.tolist())

            if motion_expl.t_articulation is not None:
                motion_params["t_articulation"].append(
                    motion_expl.t_articulation.tolist()
                )
            if motion_expl.so3 is not None:
                motion_params["joint_so3"].append(motion_expl.so3.tolist())  # K,3

        with open(f"{save_dir}/{cate}/motion.json", "w") as fp:
            json.dump(motion_params, fp)


@torch.no_grad()
def extract_motion_params(model, opts, data_info):
    # get rest mesh
    meshes_rest = model.fields.extract_canonical_meshes(
        grid_size=opts["grid_size"],
        level=opts["level"],
        inst_id=opts["inst_id"],
        vis_thresh=opts["vis_thresh"],
        use_extend_aabb=opts["extend_aabb"],
    )

    # get deformation
    motion_tuples = {}
    for cate, field in model.fields.field_params.items():
        meshes_rest[cate], motion_tuples[cate] = extract_deformation(
            field, meshes_rest[cate], opts["inst_id"], opts["flag"]
        )

    # scale
    if "bg" in model.fields.field_params.keys():
        bg_field = model.fields.field_params["bg"]
        bg_scale = bg_field.logscale.exp().cpu().numpy()
    if "fg" in model.fields.field_params.keys():
        fg_field = model.fields.field_params["fg"]
        fg_scale = fg_field.logscale.exp().cpu().numpy()

    if (
        "bg" in model.fields.field_params.keys()
        and model.fields.field_params["bg"].valid_field2world()
    ):
        # visualize ground plane
        field2world = (
            model.fields.field_params["bg"].get_field2world(opts["inst_id"]).cpu()
        )
        field2world[..., :3, 3] *= bg_scale
        meshes_rest["bg"] = append_xz_plane(
            meshes_rest["bg"], field2world.inverse(), scale=20 * bg_scale
        )

    if "fg" in model.fields.field_params.keys():
        meshes_rest["fg"] = meshes_rest["fg"].apply_scale(1.0 / fg_scale)
        rescale_motion_tuples(motion_tuples["fg"], fg_scale)
    if "bg" in model.fields.field_params.keys():
        meshes_rest["bg"] = meshes_rest["bg"].apply_scale(1.0 / bg_scale)
        rescale_motion_tuples(motion_tuples["bg"], bg_scale)
    return meshes_rest, motion_tuples


@torch.no_grad()
def export(opts, Trainer=Trainer):
    model, data_info, ref_dict = Trainer.construct_test_model(opts)
    if opts["flag"] == "":
        sub_dir = f"export_{opts['inst_id']:04d}"
    else:
        sub_dir = f"export_{opts['inst_id']:04d}_{opts['flag']}"
    save_dir = make_save_dir(opts, sub_dir=sub_dir)

    # save motion paramters
    meshes_rest, motion_tuples = extract_motion_params(model, opts, data_info)
    save_motion_params(meshes_rest, motion_tuples, save_dir)

    # save scene to world transform
    if (
        "bg" in model.fields.field_params.keys()
        and model.fields.field_params["bg"].valid_field2world()
    ):
        field2world = model.fields.field_params["bg"].get_field2world(opts["inst_id"])
        field2world = field2world.cpu().numpy().tolist()
        json.dump(field2world, open(f"{save_dir}/bg/field2world.json", "w"))

    # same raw image size and intrinsics
    with torch.no_grad():
        intrinsics = model.intrinsics.get_intrinsics(opts["inst_id"])
        camera_info = {}
        camera_info["raw_size"] = data_info["raw_size"][opts["inst_id"]].tolist()
        camera_info["intrinsics"] = intrinsics.cpu().numpy().tolist()
        json.dump(camera_info, open(f"{save_dir}/camera.json", "w"))

    # save reference images
    raw_size = data_info["raw_size"][opts["inst_id"]]  # full range of pixels
    save_rendered(ref_dict, save_dir, raw_size, data_info["apply_pca_fn"])
    print(f"Saved to {save_dir}")

    # mesh rendering
    mode = "bone,shape,boneonly"
    view = "ref,front"
    for m in mode.split(","):
        cmd = f"python lab4d/render_mesh.py --mode {m} --view {view} --testdir {save_dir}"
        print(f"Running: {cmd}")
        if signal.SIGINT == os.system(cmd):
            raise KeyboardInterrupt(cmd)


def main(_):
    opts = get_config()
    cmd = f"python lab4d/render_intermediate.py --testdir logdir/{opts['seqname']}-{opts['logname']}"
    if signal.SIGINT == os.system(cmd):
        raise KeyboardInterrupt(cmd)
    export(opts)


if __name__ == "__main__":
    app.run(main)
