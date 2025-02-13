# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
import argparse
import cv2
import glob
import json
import multiprocessing
import numpy as np
import trimesh
import tqdm


def load_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, process=False)
    return mesh


def load_mesh_part(mesh_paths_part):
    mesh_part = []
    for mesh_path in tqdm.tqdm(mesh_paths_part, desc="loading meshes"):
        mesh = load_mesh(mesh_path)
        mesh_part.append(mesh)
    return mesh_part


def load_meshes(mesh_paths, n_procs=2):
    mesh_paths_parts = [mesh_paths[i::n_procs] for i in range(n_procs)]
    if n_procs > 1:
        with multiprocessing.Pool(n_procs) as p:
            mesh_parts = p.map(load_mesh_part, mesh_paths_parts)
    else:
        mesh_parts = [load_mesh_part(mesh_paths_part) for mesh_paths_part in mesh_paths_parts]

    meshes = []
    for i in range(len(mesh_paths)):
        meshes.append(mesh_parts[i % n_procs][i // n_procs])
    return meshes


class MeshLoader:
    def __init__(self, testdir, mode, compose_mode):
        # io
        camera_info = json.load(open(f"{testdir}/camera.json", "r"))
        intrinsics = np.asarray(camera_info["intrinsics"], dtype=np.float32)
        raw_size = camera_info["raw_size"]  # h,w
        if len(glob.glob(f"{testdir}/fg/mesh/*.ply")) > 0:
            primary_dir = f"{testdir}/fg"
            secondary_dir = f"{testdir}/bg"
        else:
            primary_dir = f"{testdir}/bg"
            secondary_dir = f"{testdir}/fg"  # never use fg for secondary
        path_list = sorted([i for i in glob.glob(f"{primary_dir}/mesh/*.ply")])
        if len(path_list) == 0:
            raise ValueError(f"no mesh found that matches {primary_dir}")

        # check render mode
        if mode != "":
            pass
        elif len(glob.glob(f"{primary_dir}/bone/*")) > 0:
            mode = "bone"
        else:
            mode = "shape"

        if compose_mode != "":
            pass
        elif len(glob.glob(f"{secondary_dir}/mesh/*")) > 0:
            compose_mode = "compose"
        else:
            compose_mode = "primary"

        # get cam dict
        field2cam_fg_dict = json.load(open(f"{primary_dir}/motion.json", "r"))
        field2cam_fg_dict = field2cam_fg_dict["field2cam"]
        if compose_mode == "compose":
            field2cam_bg_dict = json.load(open(f"{secondary_dir}/motion.json", "r"))
            field2cam_bg_dict = np.asarray(field2cam_bg_dict["field2cam"])

            field2world_path = f"{testdir}/bg/field2world.json"
            field2world = np.asarray(json.load(open(field2world_path, "r")))
            world2field = np.linalg.inv(field2world)

        self.mode = mode
        self.compose_mode = compose_mode
        self.testdir = testdir
        self.intrinsics = intrinsics
        self.raw_size = raw_size
        self.path_list = path_list
        self.field2cam_fg_dict = field2cam_fg_dict
        if compose_mode == "compose":
            self.field2cam_bg_dict = field2cam_bg_dict
            self.field2world = field2world
            self.world2field = world2field
        else:
            self.field2cam_bg_dict = None
            self.field2world = None
            self.world2field = None

    def __len__(self):
        return len(self.path_list)

    def load_files(self, ghosting=False):
        mode = self.mode
        compose_mode = self.compose_mode
        path_list = self.path_list
        field2cam_fg_dict = self.field2cam_fg_dict
        field2cam_bg_dict = self.field2cam_bg_dict
        field2world = self.field2world
        world2field = self.world2field

        mesh_dict = {}
        extr_dict = {}
        bone_dict = {}
        scene_dict = {}
        ghost_dict = {}
        aabb_min = np.asarray([np.inf, np.inf])
        aabb_max = np.asarray([-np.inf, -np.inf])

        # Load meshes in parallel
        n_procs = 8
        loaded_meshes = load_meshes(path_list, n_procs=n_procs)
        if mode == "bone" or mode == "boneonly":
            bone_path_list = [mesh_path.replace("mesh", "bone") for mesh_path in path_list]
            loaded_bones = load_meshes(bone_path_list, n_procs=n_procs)
        if compose_mode == "compose":
            scene_path_list = [mesh_path.replace("fg/mesh", "bg/mesh") for mesh_path in path_list]
            loaded_scenes = load_meshes(scene_path_list, n_procs=n_procs)

        # Build output dict
        for counter, mesh_path in enumerate(tqdm.tqdm(path_list, desc="mesh_loader")):
            frame_idx = int(mesh_path.split("/")[-1].split(".")[0])
            mesh = loaded_meshes[counter]
            mesh.visual.vertex_colors = mesh.visual.vertex_colors
            field2cam_fg = np.asarray(field2cam_fg_dict[frame_idx])

            # post-modify the scale of the fg
            # mesh.vertices = mesh.vertices / 2
            # field2cam_fg[:3, 3] = field2cam_fg[:3, 3] / 2

            mesh_dict[frame_idx] = mesh
            extr_dict[frame_idx] = field2cam_fg

            if mode == "bone" or mode == "boneonly":
                # load bone
                bone = loaded_bones[counter]
                bone.visual.vertex_colors = bone.visual.vertex_colors
                bone_dict[frame_idx] = bone

            if compose_mode == "compose":
                # load scene
                scene = loaded_scenes[counter]
                scene.visual.vertex_colors = scene.visual.vertex_colors

                # align bg floor with xz plane
                scene.vertices = (
                    scene.vertices @ field2world[:3, :3].T + field2world[:3, 3]
                )
                field2cam_bg = field2cam_bg_dict[frame_idx] @ world2field
                field2cam_bg_dict[frame_idx] = field2cam_bg

                scene_dict[frame_idx] = scene
                # use scene camera
                extr_dict[frame_idx] = field2cam_bg_dict[frame_idx]
                # transform to scene
                object_to_scene = (
                    np.linalg.inv(field2cam_bg_dict[frame_idx]) @ field2cam_fg
                )
                mesh_dict[frame_idx].apply_transform(object_to_scene)
                if mode == "bone":
                    bone_dict[frame_idx].apply_transform(object_to_scene)

                if ghosting:
                    total_ghost = 10
                    ghost_skip = len(path_list) // total_ghost
                    if "ghost_list" in locals():
                        if counter % ghost_skip == 0:
                            mesh_ghost = mesh_dict[frame_idx].copy()
                            mesh_ghost.visual.vertex_colors[:, 3] = 102
                            ghost_list.append(mesh_ghost)
                    else:
                        ghost_list = [mesh_dict[frame_idx]]
                    ghost_dict[frame_idx] = [mesh.copy() for mesh in ghost_list]

            # update aabb # x,z coords
            if compose_mode == "compose":
                bounds = scene_dict[frame_idx].bounds
            else:
                bounds = mesh_dict[frame_idx].bounds
            aabb_min = np.minimum(aabb_min, bounds[0, [0, 2]])
            aabb_max = np.maximum(aabb_max, bounds[1, [0, 2]])

        self.mesh_dict = mesh_dict
        self.extr_dict = extr_dict
        self.bone_dict = bone_dict
        self.scene_dict = scene_dict
        self.ghost_dict = ghost_dict
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max

    def query_frame(self, frame_idx):
        input_dict = {}
        if self.mode == "shape":
            input_dict["shape"] = self.mesh_dict[frame_idx]
        elif self.mode == "boneonly":
            input_dict["bone"] = self.bone_dict[frame_idx]
        elif self.mode == "bone":
            input_dict["shape"] = self.mesh_dict[frame_idx]
            input_dict["bone"] = self.bone_dict[frame_idx]
            # make shape transparent and gray
            input_dict["shape"].visual.vertex_colors[:3] = 102
            input_dict["shape"].visual.vertex_colors[3:] = 128
        else:
            input_dict["shape"] = self.mesh_dict[frame_idx]

        if self.compose_mode == "compose":
            scene_mesh = self.scene_dict[frame_idx]
            scene_mesh.visual.vertex_colors[:, :3] = np.asarray([[224, 224, 54]])
            input_dict["scene"] = scene_mesh

        if len(self.ghost_dict) > 0:
            ghost_mesh = trimesh.util.concatenate(self.ghost_dict[frame_idx])
            input_dict["ghost"] = ghost_mesh

        return input_dict

    def print_info(self):
        print(f"[mode={self.mode}, compose={self.compose_mode}] rendering {len(self)} meshes from {self.testdir}")
