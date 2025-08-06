# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
import pdb
from threading import Thread

import numpy as np
import jittor as jt
from PIL import Image
from tqdm import tqdm


def get_sdpa_settings():
    if jt.flags.use_cuda and jt.compiler.has_cuda:
        old_gpu = False
        use_flash_attn = False
        math_kernel_on = True
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from sam2 import _C

    return _C.get_connected_componnets(jt.misc.contiguous(mask.uint8))


def mask_to_box(masks: jt.Var):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    xs = jt.arange(w, dtype=jt.int32)
    ys = jt.arange(h, dtype=jt.int32)
    grid_xs, grid_ys = jt.meshgrid(xs, ys)
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    _, min_xs = jt.argmin(jt.where(masks, grid_xs, w).reshape(B, 1, -1), dim=-1)
    _, max_xs = jt.argmax(jt.where(masks, grid_xs, -1).reshape(B, 1, -1), dim=-1)
    _, min_ys = jt.argmin(jt.where(masks, grid_ys, h).reshape(B, 1, -1), dim=-1)
    _, max_ys = jt.argmax(jt.where(masks, grid_ys, -1).reshape(B, 1, -1), dim=-1)
    bbox_coords = jt.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = jt.array(img_np).permute(2, 0, 1)  # transpose to [3, 1024, 1024]
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width


class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(
            self,
            img_paths,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
    ):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self.images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)

        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc="frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


def load_video_frames(
        img_paths_list,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    img_paths = img_paths_list
    num_frames = len(img_paths)
    img_mean = jt.array(img_mean, dtype=jt.float32)[:, None, None]
    img_std = jt.array(img_std, dtype=jt.float32)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(
            img_paths,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = jt.zeros((num_frames, 3, image_size, image_size), dtype="float32")
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if offload_video_to_cpu:
        jt.flags.use_cuda = 0
    # normalize by mean and std
    pdb.set_trace()
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    # Holes are those connected components in background with area <= self.max_area
    # (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"

    input_mask = mask
    try:
        labels, areas = get_connected_components((mask <= 0).float())
        is_hole = (labels > 0) & (areas <= max_area)
        # We fill holes with a small positive mask score (0.1) to change them to foreground.
        mask = jt.where(is_hole, 0.1, mask)
    except Exception as e:
        # Skip the post-processing step on removing small holes if the CUDA kernel fails
        warnings.warn(
            f"{e}\n\nSkipping the post-processing step due to the error above. You can "
            "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
            "functionality may be limited (which doesn't affect the results in most cases; see "
            "https://github.com/facebookresearch/segment-anything-2/blob/main/INSTALL.md).",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = jt.concat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = jt.concat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}


def load_video_frames_by_np_data(
        np_images,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    num_frames = np_images.shape[0]
    img_mean = jt.array(img_mean, dtype=jt.float32)[:, None, None]
    img_std = jt.array(img_std, dtype=jt.float32)[:, None, None]

    # if async_loading_frames:
    #     lazy_images = AsyncVideoFrameLoader(
    #         img_paths,
    #         image_size,
    #         offload_video_to_cpu,
    #         img_mean,
    #         img_std,
    #         compute_device,
    #     )
    #     return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = jt.zeros((num_frames, 3, image_size, image_size), dtype="float32")
    for n in range(num_frames):
        img = np_images[n]
        img_var = jt.array(img)
        img_var = img_var.unsqueeze(0)
        img_resized = jt.nn.interpolate(img_var, size=(image_size, image_size), mode='bilinear',
                                        align_corners=False)
        images[n] = img_resized.squeeze(0)
        video_height, video_width = np_images.shape[-2], np_images.shape[-1]

    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width
