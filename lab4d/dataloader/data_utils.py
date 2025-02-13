# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
import configparser
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from lab4d.utils.numpy_utils import pca_numpy


class FrameInfo:
    """Metadata about the frames in a dataset

    Args:
        ref_list (list(str)): List of paths to all filtered RGB frames in this video
    Attributes:
        num_frames (int): Number of frames after filtering out static frames.
        num_frames_raw (int): Total number of frames.
        frame_map (list(int)): Mapping from JPEGImages (filtered frames) to
          JPEGImagesRaw (all frames).
    """

    def __init__(self, ref_list):
        self.num_frames = len(ref_list)
        # need the raw frame index to apply Fourier time embedding
        raw_dir = ref_list[0].rsplit("/", 1)[0].replace("JPEGImages", "JPEGImagesRaw")
        self.num_frames_raw = len(glob.glob(raw_dir + "/*.jpg"))
        assert self.num_frames_raw > 0 
        self.frame_map = [int(path.split("/")[-1].split(".")[0]) for path in ref_list]


def train_loader(opts_dict, uncertainty_map):
    """Construct the training dataloader.

    Args:
        opts_dict (Dict): Defined in Trainer::construct_dataset_opts()
        uncertainty_map (Dict(np.array)): Uncertainty map per video
    Returns:
        dataloader (:class:`pytorch:torch.utils.data.DataLoader`): Training dataloader
    """
    # Set to 0 to debug the data loader
    num_workers = opts_dict["num_workers"]
    print(f"Num workers: {num_workers}")
    print(f"Num iterations per round: {opts_dict['iters_per_round']}*{opts_dict['grad_accum']}")
    print(f"Num image samples per iteration: {opts_dict['imgs_per_gpu'] * opts_dict['ngpu']}")
    print(f"Num pixel samples per image: {opts_dict['pixels_per_image']}")

    dataset = config_to_dataset(opts_dict, uncertainty_map=uncertainty_map)

    sampler = torch.utils.data.RandomSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=opts_dict["imgs_per_gpu"],
        num_workers=num_workers,
        drop_last=True,
        # worker_init_fn=_init_fn,
        pin_memory=True,
        sampler=sampler,
    )
    return dataloader


def eval_loader(opts_dict, uncertainty_map):
    """Construct the evaluation dataloader.

    Args:
        opts_dict (Dict): Defined in Trainer::construct_dataset_opts()
        uncertainty_map (Dict(np.array)): Uncertainty map per video
    Returns:
        dataloader (torch.utils.data.DataLoader): Evaluation dataloader
    """
    num_workers = 0

    dataset = config_to_dataset(opts_dict, is_eval=True, uncertainty_map=uncertainty_map)
    dataset = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
    )
    return dataset


def duplicate_dataset(opts, datalist, datalist_full=None, full_ratio=10):
    """Duplicate a list of per-video datasets, so that the length matches the
    desired number of iterations per round during training.

    Args:
        datalist (List(VidDataset)): A list of per-video datasets
        datalist_full (List(VidDataset)): If provided, a list of per-video
            datasets that sample from the full image
        full_ratio (int): If provided, use datalist_full for 1/X fraction of the time
    Returns:
        datalist_mul (List(VidDataset)): Duplicated dataset list
    """
    num_samp = np.sum([len(i) for i in datalist])
    if num_samp == 0:
        raise ValueError("Dataset is empty")
    dup_num = opts["iters_per_round"] * opts["grad_accum"] / (num_samp / opts["ngpu"] / opts["imgs_per_gpu"])
    dup_num = int(dup_num) + 1

    if datalist_full is None:
        datalist_mul = datalist * dup_num
    else:
        datalist_mul = []
        for idx in range(dup_num):
            if idx % full_ratio == 0:
                datalist_mul += datalist_full
            else:
                datalist_mul += datalist

    return datalist_mul


def config_to_dataset(opts, is_eval=False, gpuid=[], uncertainty_map=None):
    """Construct a PyTorch dataset that includes all videos in a sequence.

    Args:
        opts (Dict): Defined in Trainer::construct_dataset_opts()
        is_eval (bool): Unused
        gpuid (List(int)): Select a subset based on gpuid for npy generation
        uncertainty_map (Dict(np.array)): Uncertainty array for each video
    Returns:
        dataset (torch.utils.data.Dataset): Concatenation of datasets for each
            video in the sequence `opts["seqname"]`
    """
    config = configparser.RawConfigParser()
    config.read(f"database/configs/{opts['seqname']}.config")
    numvid = len(config.sections()) - 1
    datalist = []
    for vidid in range(numvid):
        img_path = str(config.get(f"data_{vidid}", "img_path"))
        vidname = img_path.strip("/").split("/")[-1]
        if uncertainty_map is None:
            uncertainty = None
        else:
            uncertainty = uncertainty_map[f"{vidname}-{opts['data_prefix']}"]
        dataset = section_to_dataset(opts, config, vidid, data_prefix=None, uncertainty=uncertainty)
        datalist.append(dataset)

    if opts["multiply"]:
        full_prefix = opts["data_prefix"].replace("crop", "full")
        datalist_full = [
            section_to_dataset(opts, config, vidid, data_prefix=full_prefix, uncertainty=None)
            for vidid in range(numvid)
        ]
        datalist = duplicate_dataset(opts, datalist, datalist_full)

    # select a subset based on gpuid (for npy generation)
    if len(gpuid) == 2:  # current id, total
        vid_per_gpu = int(np.ceil(len(datalist) / gpuid[1]))
        id_start = gpuid[0] * vid_per_gpu
        datalist = datalist[id_start : id_start + vid_per_gpu]
    dataset = torch.utils.data.ConcatDataset(datalist)
    return dataset


def section_to_dataset(opts, config, vidid, data_prefix=None, uncertainty=None):
    """Construct a PyTorch dataset for a single video in a sequence
    using `opts["dataset_constructor"]`

    Args:
        opts (Dict): Defined in Trainer::construct_dataset_opts()
        config (RawConfigParser): Config parser object
        vidid (int): Which video in the sequence
        uncertainty (np.array): Uncertainty array for the specified video
    Returns:
        dataset (torch.utils.data.Dataset): Dataset for the video
    """
    config_dict = load_config(config, "data")
    config_dict = load_config(config, f"data_{vidid}", current_dict=config_dict)

    rgblist = sorted(glob.glob(f"{config_dict['rgb_path']}/*.jpg"))
    if config_dict["end_frame"] > -1:
        rgblist = rgblist[: config_dict["end_frame"]]
    if config_dict["init_frame"] > 0:
        rgblist = rgblist[config_dict["init_frame"] :]

    dataset = opts["dataset_constructor"](
        opts,
        rgblist=rgblist,
        dataid=vidid,
        ks=config_dict["ks"],
        raw_size=config_dict["raw_size"],
        data_prefix=data_prefix,
        uncertainty=uncertainty,
    )
    return dataset


def load_config(config, dataname, current_dict=None):
    """Load a section from a .config metadata file

    Args:
        config (RawConfigParser): Config parser object
        dataname (str): Name of section to load
        currect_dict (Dict): If given, load into an existing dict. Otherwise
            return a new dict
    """
    if current_dict is None:
        config_dict = {}
    else:
        config_dict = current_dict

    try:
        config_dict["rgb_path"] = str(config.get(dataname, "img_path"))
    except:
        pass

    try:
        config_dict["init_frame"] = int(config.get(dataname, "init_frame"))
    except:
        pass

    try:
        config_dict["end_frame"] = int(config.get(dataname, "end_frame"))
    except:
        pass

    try:
        config_dict["ks"] = [float(i) for i in config.get(dataname, "ks").split(" ")]
    except:
        pass

    try:
        config_dict["raw_size"] = [
            int(i) for i in config.get(dataname, "shape").split(" ")
        ]
    except:
        pass

    return config_dict


def get_data_info(loader):
    """Extract dataset metadata from a dataloader

    Args:
        loader (torch.utils.data.DataLoader): Evaluation dataloader
    Returns:
        data_info (Dict): Dataset metadata
    """
    data_info = {}
    dataset_list = loader.dataset.datasets
    frame_offset = [0]
    frame_offset_raw = [0]
    frame_mapping = []
    intrinsics = []
    raw_size = []
    feature_pxs = []
    motion_scales = []

    for dataset in dataset_list:
        frame_info = FrameInfo(dataset.dict_list["ref"])
        frame_offset.append(frame_info.num_frames)
        frame_offset_raw.append(frame_info.num_frames_raw)
        frame_mapping += [
            i + np.sum(frame_offset_raw[:-1]) for i in frame_info.frame_map
        ]
        intrinsics += [dataset.ks] * frame_info.num_frames
        raw_size += [dataset.raw_size]

        feature_array = dataset.mmap_list["feature"].reshape(-1, 16)
        feature_array = feature_array[np.linalg.norm(feature_array, 2, -1) > 0]

        # # randomly sample 1k non-zero pixels for PCA per-video:
        # feature_1k = feature_array[np.random.rand(*feature_array.shape[:-1]) < 0.001]
        # feature_pxs.append(feature_1k)

        # sampling a fixed set of pixels for PCA:
        feature_pxs.append(feature_array)

        # compute motion magnitude
        mask = dataset.mmap_list["mask"][:-1, ..., 0].copy()
        if dataset.field_type == "bg":
            mask = np.logical_not(mask)
        elif dataset.field_type == "fg":
            pass
        elif dataset.field_type == "comp":
            mask[:] = True
        else:
            raise ValueError(f"Unknown field type: {dataset.field_type}")
        flow = dataset.mmap_list["flowfw"][1][mask, :2]
        motion_scale = np.linalg.norm(flow, 2, -1).mean()
        motion_scales.append(motion_scale)

    # compute PCA on non-zero features
    feature_pxs = np.concatenate(feature_pxs, 0)
    data_info["apply_pca_fn"] = pca_numpy(feature_pxs, n_components=3)

    # store motion magnitude
    data_info["motion_scales"] = motion_scales
    # print("motion scales: ", motion_scales)

    frame_info = {}
    frame_info["frame_offset"] = np.asarray(frame_offset).cumsum()
    frame_info["frame_offset_raw"] = np.asarray(frame_offset_raw).cumsum()
    frame_info["frame_mapping"] = frame_mapping
    data_info["frame_info"] = frame_info

    data_info["total_frames"] = frame_info["frame_offset"][-1]
    data_info["intrinsics"] = np.asarray(intrinsics)  # N,4
    data_info["raw_size"] = np.asarray(raw_size)  # M,2

    data_path_dict = merge_dict_list(loader)
    data_info.update(load_small_files(data_path_dict))
    return data_info, data_path_dict


def merge_dict_list(loader):
    """For a sequence of videos, construct a dict .npy/.txt paths that contain
    all the frame data and annotations from the whole sequence

    Args:
        loader (torch.utils.data.DataLoader): Dataloader for a video sequence
    Returns:
        dict_list (Dict(str, List(str))): Maps each frame/annotation type to a
            list of .npy/.txt paths for that type
    """
    dataset_list = loader.dataset.datasets
    data_dict = {}
    for dataset in dataset_list:
        for k, path_list in dataset.dict_list.items():
            if k not in data_dict:
                data_dict[k] = []
            if isinstance(path_list, str):
                data_dict[k].append(path_list)
            else:
                data_dict[k] += path_list
    return data_dict


def load_small_files(data_path_dict):
    """For a sequence of videos, load small dataset files into memory

    Args:
        data_path_dict (Dict(str, List(str))): Maps each annotation type to a
            list of .npy/.txt paths for that type
    Returns:
        data_info (Dict): Dataset metadata
    """
    data_info = {}
    # data_info["crop2raw"] = np.concatenate(
    #     [np.load(path).astype(np.float32) for path in data_path_dict["crop2raw"]], 0
    # )  # N,4

    # bg/fg camera
    rtmat_bg = []
    for vid, path in enumerate(data_path_dict["cambg"]):
        # get N
        num_frames = np.load(data_path_dict["is_detected"][vid]).shape[0]
        if os.path.exists(path):
            rtmat_bg.append(np.load(path).astype(np.float32))
        else:
            rtmat_bg.append(np.eye(4)[None].repeat(num_frames, 0))
            print(f"Warning: no bg camera found at {path}")
    rtmat_bg = np.concatenate(rtmat_bg, 0)  # N,4,4

    rtmat_fg = []
    for vid, path in enumerate(data_path_dict["camfg"]):
        # get N
        num_frames = np.load(data_path_dict["is_detected"][vid]).shape[0]
        if os.path.exists(path):
            rtmat_fg.append(np.load(path).astype(np.float32))
        else:
            rtmat_fg.append(np.eye(4)[None].repeat(num_frames, 0))
            print(f"Warning: no fg camera found at {path}")

    rtmat_fg = np.concatenate(rtmat_fg, 0)

    # hard-code for now
    vis_info = {"bg": 0, "fg": 1}  # video instance segmentation info
    data_info["vis_info"] = vis_info
    data_info["rtmat"] = np.stack([rtmat_bg, rtmat_fg], 0)
    try:
        data_info["joint_angles"] = np.concatenate(
            [np.load(path).astype(np.float32) for path in data_path_dict["joint_angles"]], 0
        )  # N, 3*23
    except:
        pass

    # path to centered mesh files
    geom_path_bg = []
    geom_path_fg = []
    for path in data_path_dict["cambg"]:
        camera_prefix = path.rsplit("/", 1)[0]
        geom_path_bg.append(f"{camera_prefix}/mesh-00-centered.ply")
        geom_path_fg.append(f"{camera_prefix}/mesh-01-centered.ply")
    data_info["geom_path"] = [geom_path_bg, geom_path_fg]
    return data_info


def get_vid_length(inst_id, data_info):
    """Compute the length of a video

    Args:
        inst_id (int): Video to check
        data_info (Dict): Dataset metadata
    """
    frame_offset_raw = data_info["frame_info"]["frame_offset_raw"]
    vid_length = frame_offset_raw[1:] - frame_offset_raw[:-1]
    vid_length = vid_length[inst_id]
    return vid_length
