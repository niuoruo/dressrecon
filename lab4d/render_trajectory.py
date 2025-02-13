# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
# python lab4d/render_mesh.py --testdir logdir//ama-bouncing-4v-ppr/export_0000/ --view bev --ghosting
import argparse
import cv2
import glob
import json
import multiprocessing
import numpy as np
import os
import sys
import trimesh
import tqdm


cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from lab4d.utils.io import save_vid
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.mesh_loader import MeshLoader


def segment_to_mesh(prev_traj_pts, next_traj_pts, colors):
    segment_meshes = []
    segment_colors = []
    for seg, color in zip(np.stack([prev_traj_pts, next_traj_pts], axis=1), colors):
        seg_cylinder = trimesh.creation.cylinder(radius=0.002, sections=3, segment=seg)
        seg_color = np.tile(color[None], (seg_cylinder.vertices.shape[0], 1))
        segment_meshes.append(seg_cylinder)
        segment_colors.append(seg_color)

    segment_mesh = trimesh.util.concatenate(segment_meshes)
    segment_mesh.visual.vertex_colors = np.concatenate(segment_colors, axis=0)
    return segment_mesh


def segment_to_mesh_part(args_part, trajectories, segment_colors):
    segment_mesh_parts = []
    for frame_idx in tqdm.tqdm(args_part, desc="computing trajectory meshes"):
        prev_traj_pts = trajectories[frame_idx]
        next_traj_pts = trajectories[frame_idx + 1]
        segment_mesh = segment_to_mesh(prev_traj_pts, next_traj_pts, segment_colors)
        segment_mesh_parts.append(segment_mesh)
    return segment_mesh_parts


def render_frame(frame_idx, input_dict, cam_extr, cam_intr, segment_meshes, renderer, window_size=15):
    # concatenate trajectory from line segments
    if frame_idx > 0:
        trajectory_meshes = []
        for frame_idx_prev in range(max(0, frame_idx - window_size), frame_idx):
            trajectory_meshes.append(segment_meshes[frame_idx_prev])
        input_dict["trajectory"] = trimesh.util.concatenate(trajectory_meshes)

    renderer.set_camera(cam_extr)
    renderer.set_intrinsics(cam_intr)
    renderer.align_light_to_camera()

    color = renderer.render(input_dict)[0]
    color = color.astype(np.uint8)
    # Avoid putting text for eccv supp visuals
    # color = cv2.putText(
    #     color, f"frame: {frame_idx:02d}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (256, 0, 0), 2,
    # )

    if frame_idx > 0:
        del input_dict["trajectory"]

    return color


def render_frame_part(input_args_part, segment_meshes, raw_size):
    renderer = PyRenderWrapper(raw_size)
    frames_part = []
    for frame_idx, input_dict, cam_extr, cam_intr in tqdm.tqdm(input_args_part, desc="rendering frames"):
        frame = render_frame(frame_idx, input_dict, cam_extr, cam_intr, segment_meshes, renderer)
        frames_part.append(frame)
    return frames_part


def main(args):
    loader = MeshLoader(args.testdir, args.mode, args.compose_mode)
    loader.print_info()
    loader.load_files(ghosting=args.ghosting)
    raw_size = loader.raw_size
    n_procs = 1

    # extract 3D trajectories
    trajectories = []
    vert_idxs = None
    for frame_idx, mesh_obj in tqdm.tqdm(loader.mesh_dict.items()):
        # Reset mesh colors to gray
        mesh_obj.visual.vertex_colors[:] = (102, 102, 102, 255)

        if vert_idxs is None:
            vert_idxs = np.random.choice(mesh_obj.vertices.shape[0], 2000, replace=False)
        trajectories.append(mesh_obj.vertices[vert_idxs])

    # set the segment color to XYZ location at first frame
    segment_colors = trajectories[0]  # N_verts, 3
    segment_colors = (
        (segment_colors - segment_colors.min(axis=0)) / (segment_colors.max(axis=0) - segment_colors.min(axis=0))
    )  # [0, 1]
    segment_colors = (255 * segment_colors).astype(np.uint8)

    # render
    for view in args.view.split(","):
        print(f"Rendering [{view}]:")

        cam_extrs = [None for i in range(len(loader.mesh_dict))]
        cam_intrs = [None for i in range(len(loader.mesh_dict))]
        for frame_idx in range(len(loader.mesh_dict)):
            if loader.compose_mode == "primary":
                if view == "ref":
                    # set camera extrinsics
                    cam_extrs[frame_idx] = loader.extr_dict[frame_idx]
                    # set camera intrinsics
                    cam_intrs[frame_idx] = loader.intrinsics[frame_idx]
                elif view == "bev":
                    raise NotImplementedError
                elif view == "front":
                    # set camera extrinsics
                    cam_extrs[frame_idx] = loader.extr_dict[0]
                    # set camera intrinsics
                    cam_intrs[frame_idx] = loader.intrinsics[0]

            else:
                raise NotImplementedError

        # compute segment meshes in parallel
        segment_args = []
        for frame_idx in range(len(loader.mesh_dict) - 1):
            segment_args.append(frame_idx)
        segment_args_parts = [segment_args[i::n_procs] for i in range(n_procs)]
        args_parts = [(segment_args_parts[i], trajectories, segment_colors) for i in range(n_procs)]

        if n_procs > 1:
            with multiprocessing.Pool(n_procs) as p:
                segment_meshes_parts = p.starmap(segment_to_mesh_part, args_parts)
        else:
            segment_meshes_parts = [segment_to_mesh_part(*args_part) for args_part in args_parts]

        segment_meshes = []
        for i in range(len(loader.mesh_dict) - 1):
            segment_meshes.append(segment_meshes_parts[i % n_procs][i // n_procs])

        # render frames in parallel
        input_args = [
            (frame_idx, loader.query_frame(frame_idx), cam_extrs[frame_idx], cam_intrs[frame_idx])
            for frame_idx in range(len(loader.mesh_dict))
        ]
        input_args_parts = [input_args[i::n_procs] for i in range(n_procs)]
        args_parts = [(input_args_parts[i], segment_meshes, raw_size) for i in range(n_procs)]

        if n_procs > 1:
            with multiprocessing.Pool(n_procs) as p:
                frames_parts = p.starmap(render_frame_part, args_parts)
        else:
            frames_parts = [render_frame_part(*args_part) for args_part in args_parts]

        frames = []
        for i in range(len(loader.mesh_dict) - 1):
            frames.append(frames_parts[i % n_procs][i // n_procs])
        
        save_path = f"{args.testdir}/trajectory-{loader.mode}-{loader.compose_mode}-{view}"
        save_vid(save_path, frames, suffix=".mp4", upsample_frame=-1, fps=args.fps)
        print(f"saved to {save_path}.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to render extraced meshes")
    parser.add_argument("--testdir", default="", help="path to the directory with results")
    parser.add_argument("--fps", default=30, type=int, help="fps of the video")
    parser.add_argument("--mode", default="", type=str, help="{shape, bone, boneonly}")
    parser.add_argument("--compose_mode", default="", type=str, help="{object, scene}")
    parser.add_argument("--ghosting", action="store_true", help="ghosting")
    parser.add_argument("--view", default="ref", type=str, help="{ref, bev, front}")
    args = parser.parse_args()

    main(args)
