"""
Code ported from https://github.com/cwkx/GON/blob/master/GON.py
"""
import os
import pathlib
import math
from typing import Union, Optional, List, Tuple, Text, BinaryIO
from PIL import Image

import numpy as np
import tensorflow as tf

irange = range


def create_dirs(dirpath: str):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


# helper functions
def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [tf.linspace(-1., 1., num=sidelen)])
    mgrid = tf.stack(tf.meshgrid(*tensors), axis=-1)
    mgrid = tf.reshape(mgrid, [-1, dim])
    return mgrid


def slerp(a, b, t):
    omega = tf.acos(tf.reduce_sum(a / tf.norm(a, axis=1, keepdims=True) * b / tf.norm(b, axis=1, keepdims=True), axis=1))
    omega = tf.expand_dims(omega, axis=1)
    res = (tf.sin((1.0 - t) * omega) / tf.sin(omega)) * a + (tf.sin(t * omega) / tf.sin(omega)) * b
    return res


def slerp_batch(model, z, coords, batch_size):
    # z: [B, 1, Z]
    # coords = [B, rows * cols, 2]
    lz = tf.squeeze(tf.identity(z), axis=1)
    col_size = int(np.sqrt(z.shape[0]))
    src_z = tf.tile(lz[:col_size], (col_size, 1))
    z1, z2 = tf.split(lz, 2)
    tgt_z = tf.concat([z2, z1], axis=0)
    tgt_z = tf.tile(tgt_z[:col_size], (col_size, 1))

    t = tf.linspace(0., 1., num=col_size)
    t = tf.expand_dims(t, axis=1)
    t = tf.tile(t, [1, col_size])
    t = tf.reshape(t, [batch_size, 1])

    z_slerp = slerp(src_z, tgt_z, t)
    z_slerp_rep = tf.expand_dims(z_slerp, axis=1)
    z_slerp_rep = tf.tile(z_slerp_rep, [1, coords.shape[1], 1])
    g_slerp = model(tf.concat((coords, z_slerp_rep), axis=-1))
    return g_slerp


def gon_sample(model, recent_zs, coords, batch_size):
    zs = tf.squeeze(tf.concat(recent_zs, axis=0), axis=1).numpy()
    mean = np.mean(zs, axis=0)
    cov = np.cov(zs.T)
    sample = np.random.multivariate_normal(mean, cov, size=batch_size)
    sample = tf.convert_to_tensor(sample)
    sample = tf.expand_dims(sample, axis=1)
    sample = tf.tile(sample, [1, coords.shape[1], 1])
    sample = tf.cast(sample, tf.float32)

    model_input = tf.concat((coords, sample), axis=-1)
    return model(model_input)


# Ported from pytorch utils
def make_grid(
        tensor: Union[tf.Tensor, List[tf.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
) -> tf.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (tf.is_tensor(tensor) or
            (isinstance(tensor, list) and all(tf.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = tf.stack(tensor, axis=0)

    if len(tensor.shape) == 2:  # single image H x W
        tensor = tf.expand_dims(tensor, axis=0)

    if len(tensor.shape) == 3:  # single image
        if tensor.shape[-1] == 1:  # if single-channel, convert to 3-channel
            tensor = tf.concat((tensor, tensor, tensor), axis=-1)
        tensor = tf.expand_dims(tensor, axis=0)

    if len(tensor.shape) == 4 and tensor.shape[-1] == 1:  # single-channel images
        tensor = tf.concat((tensor, tensor, tensor), axis=-1)

    if normalize is True:
        tensor = tf.identity(tensor)  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img = tf.clip_by_value(img, clip_value_min=min, clip_value_max=max)
            img = (img - min) / (max - min + 1e-5)
            return img

        def norm_range(t, range):
            if range is not None:
                t = norm_ip(t, range[0], range[1])
            else:
                t = norm_ip(t, float(tf.reduce_min(t)), float(tf.reduce_max(t)))
            return t

        if scale_each is True:
            for tix in irange(tensor.shape[0]):  # loop over mini-batch dimension
                tensor[tix] = norm_range(tensor[tix], range)
        else:
            tensor = norm_range(tensor, range)

    if tensor.shape[0] == 1:
        return tf.squeeze(tensor, axis=0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    num_channels = tensor.shape[-1]

    tensor = tensor.numpy()
    grid = tf.fill((height * ymaps + padding, width * xmaps + padding, num_channels), pad_value).numpy().astype(tensor.dtype)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid[y * height + padding: y * height + height, x * width + padding: x * width + width] += tensor[k, :, :, :]
            k = k + 1
    return grid


def save_image(
        tensor: Union[tf.Tensor, List[tf.Tensor]],
        fp: Union[Text, pathlib.Path, BinaryIO],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        format: Optional[str] = None,
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = tf.clip_by_value(grid * 255.0, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
