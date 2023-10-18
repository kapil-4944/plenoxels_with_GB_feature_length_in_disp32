import numpy as np
import logging as log
# import time
import os
import glob
# from PIL import Image
# from numpy import ndarray
from torch import Tensor

from .ray_utils import (
    generate_spherical_poses, create_meshgrid, stack_camera_dirs, get_rays, generate_spiral_path
)

import imageio.v3 as iio
#from collections import defaultdict
from typing import Dict, MutableMapping, Union, Any, List,Optional, Tuple
#import random
# import pandas as pd
import torch
import torch.utils.data
from plenoxels_with_GB_feature_length_in_disp32.intrinsics import Intrinsics
from plenoxels_with_GB_feature_length_in_disp32.llff_dataset import load_llff_poses_helper
from plenoxels_with_GB_feature_length_in_disp32.base_dataset import BaseDataset
def load_llffvideo_poses(datadir: str,
                         downsample: float,
                         split: str,
                         near_scaling: float,data_part =2) : #-> tuple[Tensor, Tensor, Intrinsics, Any, ndarray]
    """Load poses and metadata for LLFF video.

    Args:
        datadir (str): Directory containing the videos and pose information
        downsample (float): How much to downsample videos. The default for LLFF videos is 2.0
        split (str): 'train' or 'test'.
        near_scaling (float): How much to scale the near bound of poses.

    Returns:
        Tensor: A tensor of size [N, 4, 4] containing c2w poses for each camera.
        Tensor: A tensor of size [N, 2] containing near, far bounds for each camera.
        Intrinsics: The camera intrinsics. These are the same for every camera.
        List[str]: List of length N containing the path to each camera's data.
    """
    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)
    # poses, near_fars, intrinsics = poses, near_fars, intrinsics

    videopaths = np.array(glob.glob(os.path.join(datadir, '*.mp4')))

    assert poses.shape[0] == len(videopaths), \
        'Mismatch between number of cameras and number of poses!'
    videopaths.sort()

    # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
    if split == 'train':
        split_ids = np.arange(1, poses.shape[0], 7)
    elif split == 'test':
        split_ids = np.array([0])
    else:
        split_ids = np.arange(poses.shape[0])

    poses = torch.from_numpy(poses[split_ids])
    near_fars = torch.from_numpy(near_fars[split_ids])
    videopaths = videopaths[split_ids].tolist()

    return poses, near_fars, intrinsics, videopaths, split_ids


def _load_video_1cam(idx: int,
                     paths: List[str],
                     poses: torch.Tensor,
                     out_h: int,
                     out_w: int,
                     load_every: int = 1
                     ):  # -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
    filters = [
        ("scale", f"w={out_w}:h={out_h}")
    ]
    all_frames = iio.imread(
        paths[idx], plugin='pyav', format='rgb24', constant_framerate=True, thread_count=2,
        filter_sequence=filters,)
    imgs, timestamps = [], []
    for frame_idx, frame in enumerate(all_frames):
        if frame_idx % load_every != 0:
            continue
        if frame_idx >= 300:  # Only look at the first 10 seconds
            break
        # Frame is np.ndarray in uint8 dtype (H, W, C)
        imgs.append(
            torch.from_numpy(frame)
        )
        timestamps.append(frame_idx)
    imgs = torch.stack(imgs, 0)
    med_img, _ = torch.median(imgs, dim=0)  # [h, w, 3]
    return \
        (imgs,
            poses[idx].expand(len(timestamps), -1, -1),
            med_img,
            torch.tensor(timestamps, dtype=torch.int32))


def parallel_load_images(tqdm_title,
                         dset_type: str,
                         num_images: int,
                         **kwargs) -> List[Any]:
    max_threads = 10

    if  dset_type == 'video':
        fn = 1
        max_threads =4
    else:
        raise ValueError(dset_type)
    outputs = []
    if fn == 1:
        for i in range(num_images):
            fn = _load_video_1cam(idx=i, **kwargs)
            if i is not None:
                outputs.append(fn)

    return outputs


def load_llffvideo_data(videopaths: List[str],
                        cam_poses: torch.Tensor,
                        intrinsics: Intrinsics,
                        split: str,
                        keyframes: bool,
                        keyframes_take_each: Optional[int] = None,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if keyframes and (keyframes_take_each is None or keyframes_take_each < 1):
        raise ValueError(f"'keyframes_take_each' must be a positive number, "
                         f"but is {keyframes_take_each}.")

    loaded = parallel_load_images(
            dset_type="video",
            tqdm_title=f"Loading {split} data",
            num_images=len(videopaths),#%2
            paths=videopaths,#[:len(videopaths)%2],
            poses=cam_poses,
            out_h=intrinsics.height,
            out_w=intrinsics.width,
            load_every=keyframes_take_each if keyframes else 1,
        )

    imgs, poses, median_imgs, timestamps = zip(*loaded)

    timestamps = torch.cat(timestamps, 0)  # [N]
    poses = torch.cat(poses, 0)            # [N, 3, 4]
    imgs = torch.cat(imgs, 0)              # [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]

    return poses, imgs, timestamps, median_imgs


class Video360Dataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 keyframes: bool = False,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 isg: bool = False,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 ndc_far: float = 2.6):
            self.keyframes = keyframes
            self.max_cameras = max_cameras
            self.max_tsteps = max_tsteps
            self.downsample = downsample
            self.isg = isg
            self.ist = False
            # self.lookup_time = False
            self.per_cam_near_fars = None
            self.global_translation = torch.tensor([0, 0, 0])
            self.global_scale = torch.tensor([1, 1, 1])
            self.near_scaling = near_scaling
            self.ndc_far = ndc_far
            self.median_imgs = None
            self.per_cam_poses = None



            if "lego" in datadir or "dnerf" in datadir:
                dset_type = "synthetic"

            else:
                dset_type = "llff"

            if dset_type == "llff":
                if split == "render":
                    assert ndc, "Unable to generate render poses without ndc: don't know near-far."
                    per_cam_poses, per_cam_near_fars, intrinsics, _ ,video_num = load_llffvideo_poses(
                        datadir, downsample=self.downsample, split='all', near_scaling=self.near_scaling)
                    render_poses = generate_spiral_path(
                        per_cam_poses.numpy(), per_cam_near_fars.numpy(), n_frames=300,
                        n_rots=2, zrate=0.5, dt=self.near_scaling, percentile=60)
                    self.per_cam_poses = per_cam_poses.float()
                    self.video_num = video_num
                    self.poses = torch.from_numpy(render_poses).float()
                    self.per_cam_near_fars = torch.tensor([[0.4, self.ndc_far]])
                    timestamps = torch.linspace(0, 299, len(self.poses))
                    imgs = None

                else:
                    per_cam_poses, per_cam_near_fars, intrinsics, videopaths, video_num = load_llffvideo_poses(
                        datadir, downsample=self.downsample, split=split, near_scaling=self.near_scaling)
                    if split == 'test':
                        keyframes = False
                    poses, imgs, timestamps, self.median_imgs = load_llffvideo_data(
                        videopaths=videopaths, cam_poses=per_cam_poses, intrinsics=intrinsics,
                        split=split, keyframes=keyframes, keyframes_take_each=30)

                    self.poses = poses.float()
                    self.per_cam_poses = per_cam_poses.float()
                    self.video_num = video_num

                    if contraction:
                        self.per_cam_near_fars = per_cam_near_fars.float()
                    else:
                        self.per_cam_near_fars = torch.tensor(
                            [[0.0, self.ndc_far]]).repeat(per_cam_near_fars.shape[0], 1)

                # These values are tuned for the salmon video
                self.global_translation = torch.tensor([0, 0, 2.])
                self.global_scale = torch.tensor([0.5, 0.6, 1])
                # Normalize timestamps between -1, 1
                self.timestamps_max = timestamps.max()
                timestamps = (timestamps.float() / self.timestamps_max) * 2 - 1

            self.timestamps = timestamps
            self.intrinsics = intrinsics
            if split == 'train':
                    self.timestamps = self.timestamps[:, None, None].repeat(
                        1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]

            assert self.timestamps.min() >= -1.0 and self.timestamps.max() <= 1.0, "timestamps out of range."

            if imgs is not None and imgs.dtype != torch.uint8:
                    imgs = (imgs * 255).to(torch.uint8)

            if self.median_imgs is not None and self.median_imgs.dtype != torch.uint8:
                    self.median_imgs = (self.median_imgs * 255).to(torch.uint8)

            if split == 'train':
                    imgs = imgs.view(-1, imgs.shape[-1])

            elif imgs is not None:
                    imgs = imgs.view(-1, intrinsics.height * intrinsics.width, imgs.shape[-1])

                # ISG/IST weights are computed on 4x subsampled data.
            weights_subsampled = int(4 / downsample)
            if scene_bbox is not None:
                    scene_bbox = torch.tensor(scene_bbox)

            # else:
            #         scene_bbox = get_bbox(datadir, is_contracted=contraction, dset_type=dset_type)
            super().__init__(
                datadir=datadir,
                split=split,
                batch_size=batch_size,
                is_ndc=ndc,
                is_contracted=contraction,
                scene_bbox=scene_bbox,
                rays_o=None,
                rays_d=None,
                intrinsics=intrinsics,
                imgs=imgs,
                sampling_weights=None,  # Start without importance sampling, by default
                weights_subsampled=weights_subsampled,
            )

            self.isg_weights = None
            self.ist_weights = None
            if split == "train" and dset_type == 'llff':  # Only use importance sampling with DyNeRF videos
                if os.path.exists(os.path.join(datadir, f"isg_weights.pt")):
                    self.isg_weights = torch.load(os.path.join(datadir, f"isg_weights.pt"))
                    log.info(f"Reloaded {self.isg_weights.shape[0]} ISG weights from file.")
                # else:
                #     # Precompute ISG weights
                #     t_s = time.time()
                #     gamma = 1e-3 if self.keyframes else 2e-2
                #     self.isg_weights = dynerf_isg_weight(
                #         imgs.view(-1, intrinsics.height, intrinsics.width, imgs.shape[-1]),
                #         median_imgs=self.median_imgs, gamma=gamma)
                #     # Normalize into a probability distribution, to speed up sampling
                #     self.isg_weights = (self.isg_weights.reshape(-1) / torch.sum(self.isg_weights))
                #     torch.save(self.isg_weights, os.path.join(datadir, f"isg_weights.pt"))
                #     t_e = time.time()
                #     log.info(f"Computed {self.isg_weights.shape[0]} ISG weights in {t_e - t_s:.2f}s.")
                #
                # if os.path.exists(os.path.join(datadir, f"ist_weights.pt")):
                #     self.ist_weights = torch.load(os.path.join(datadir, f"ist_weights.pt"))
                #     log.info(f"Reloaded {self.ist_weights.shape[0]} IST weights from file.")
                # else:
                #     # Precompute IST weights
                #     t_s = time.time()
                #     self.ist_weights = dynerf_ist_weight(
                #         imgs.view(-1, self.img_h, self.img_w, imgs.shape[-1]),
                #         num_cameras=self.median_imgs.shape[0])
                #     # Normalize into a probability distribution, to speed up sampling
                #     self.ist_weights = (self.ist_weights.reshape(-1) / torch.sum(self.ist_weights))
                #     torch.save(self.ist_weights, os.path.join(datadir, f"ist_weights.pt"))
                #     t_e = time.time()
                #     log.info(f"Computed {self.ist_weights.shape[0]} IST weights in {t_e - t_s:.2f}s.")

            if self.isg:
                self.enable_isg()

            log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                     f"Loaded {self.split} set from {self.datadir}: "
                     f"{len(self.poses)} images of size {self.img_h}x{self.img_w}. "
                     f"Images loaded: {self.imgs is not None}. "
                     f"{len(torch.unique(timestamps))} timestamps. Near-far: {self.per_cam_near_fars}. "
                     f"ISG={self.isg}, IST={self.ist}, weights_subsampled={self.weights_subsampled}. "
                     f"Sampling without replacement={self.use_permutation}. {intrinsics}")

    def enable_isg(self):
        self.isg = True
        self.ist = False
        self.sampling_weights = self.isg_weights
        log.info(f"Enabled ISG weights.")

    def switch_isg2ist(self):
        self.isg = False
        self.ist = True
        self.sampling_weights = self.ist_weights
        log.info(f"Switched from ISG to IST weights.")

    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        if self.split == 'train':
            index = self.get_rand_ids(index)  # [batch_size // (weights_subsampled**2)]
            if self.weights_subsampled == 1 or self.sampling_weights is None:
                # Nothing special to do, either weights_subsampled = 1, or not using weights.
                image_id = torch.div(index, h * w, rounding_mode='floor')
                y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
                x = torch.remainder(index, h * w).remainder(w)
            # else:
            #     # We must deal with the fact that ISG/IST weights are computed on a dataset with
            #     # different 'downsampling' factor. E.g. if the weights were computed on 4x
            #     # downsampled data and the current dataset is 2x downsampled, `weights_subsampled`
            #     # will be 4 / 2 = 2.
            #     # Split each subsampled index into its 16 components in 2D.
            #     hsub, wsub = h // self.weights_subsampled, w // self.weights_subsampled
            #     image_id = torch.div(index, hsub * wsub, rounding_mode='floor')
            #     ysub = torch.remainder(index, hsub * wsub).div(wsub, rounding_mode='floor')
            #     xsub = torch.remainder(index, hsub * wsub).remainder(wsub)
            #     # xsub, ysub is the first point in the 4x4 square of finely sampled points
            #     x, y = [], []
            #     for ah in range(self.weights_subsampled):
            #         for aw in range(self.weights_subsampled):
            #             x.append(xsub * self.weights_subsampled + aw)
            #             y.append(ysub * self.weights_subsampled + ah)
            #     x = torch.cat(x)
            #     y = torch.cat(y)
            #     image_id = image_id.repeat(self.weights_subsampled ** 2)
            #     # Inverse of the process to get x, y from index. image_id stays the same.
            #     index = x + y * w + image_id * h * w
            x, y = x + 0.5, y + 0.5
        else:
            image_id = [index]
            x, y = create_meshgrid(height=h, width=w, dev=dev, add_half=True, flat=True)

        out = {
            "timestamps": self.timestamps[index],  # (num_rays or 1, )
            "imgs": None,

        }

        if self.split == 'train':
            num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars) * h * w)
            camera_id = torch.div(image_id, num_frames_per_camera, rounding_mode='floor')  # (num_rays)
            out['near_fars'] = self.per_cam_near_fars[camera_id, :]
        else:
            out['near_fars'] = self.per_cam_near_fars  # Only one test camera

        if self.imgs is not None:
            out['imgs'] = (self.imgs[index] / 255.0).view(-1, self.imgs.shape[-1])

        # out['video_num'] = self.video_num
        c2w = self.poses[image_id]  # [num_rays or 1, 3, 4]     CAMERA TO WORLD MATRIX
        camera_dirs = stack_camera_dirs(x, y, self.intrinsics, True)  # [num_rays, 3]
        out['rays_o'], out['rays_d'] = get_rays(
            camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics,
            normalize_rd=True)  # [num_rays, 3]

        n_c2w = self.per_cam_poses
        ones = torch.tensor([0, 0, 0, 1])
        ones = ones[None, ...]
        c2w = torch.cat((n_c2w, ones.repeat(n_c2w.shape[0], 1, 1)), dim=1)

        w2c = torch.linalg.inv(c2w)
        out['w2c'] = w2c

        imgs = out['imgs']
        # Decide BG color
        bg_color = torch.ones((1, 3), dtype=torch.float32, device=dev)
        if self.split == 'train' and imgs.shape[-1] == 4:
            bg_color = torch.rand((1, 3), dtype=torch.float32, device=dev)
        out['bg_color'] = bg_color
        # Alpha compositing
        if imgs is not None and imgs.shape[-1] == 4:
            imgs = imgs[:, :3] * imgs[:, 3:] + bg_color * (1.0 - imgs[:, 3:])
        out['imgs'] = imgs

        return out
