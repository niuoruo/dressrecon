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


def render_frame(input_dict, cam_extr, cam_intr, renderer):
    renderer.set_camera(cam_extr)
    renderer.set_intrinsics(cam_intr)
    renderer.align_light_to_camera()

    color = renderer.render(input_dict)[0]
    color = color.astype(np.uint8)
    return color


def render_frame_part(input_args_part, raw_size, mode):
    renderer = PyRenderWrapper(raw_size)
    frames_part = []
    for input_dict, cam_extr, cam_intr in tqdm.tqdm(input_args_part, desc="rendering frames"):
        frame = render_frame(input_dict, cam_extr, cam_intr, renderer)
        if mode == "shape":
            # Post-process to brighten the image for better appearance
            frame = np.clip(1.25 * frame.astype(np.float32), 0, 255).astype(np.uint8)
        frames_part.append(frame)
    return frames_part


def main(args):
    for mode in args.mode.split(","):
        loader = MeshLoader(args.testdir, mode, args.compose_mode)
        loader.print_info()
        loader.load_files(ghosting=args.ghosting)
        raw_size = loader.raw_size
        n_procs = 8

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

            # render frames in parallel
            input_args = [
                (loader.query_frame(frame_idx), cam_extrs[frame_idx], cam_intrs[frame_idx])
                for frame_idx in range(len(loader.mesh_dict))
            ]
            input_args_parts = [input_args[i::n_procs] for i in range(n_procs)]
            args_parts = [(input_args_parts[i], raw_size, loader.mode) for i in range(n_procs)]

            if n_procs > 1:
                with multiprocessing.Pool(n_procs) as p:
                    frames_parts = p.starmap(render_frame_part, args_parts)
            else:
                frames_parts = [render_frame_part(*args_part) for args_part in args_parts]

            frames = []
            for i in range(len(input_args)):
                frames.append(frames_parts[i % n_procs][i // n_procs])

            save_path = f"{args.testdir}/render-{loader.mode}-{loader.compose_mode}-{view}"
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
