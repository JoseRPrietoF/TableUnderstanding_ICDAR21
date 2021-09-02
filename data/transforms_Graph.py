from __future__ import print_function
from __future__ import division
import torch_geometric.transforms as T

# import os
import math

import numpy as np
import torch

# from torch.utils.data import Dataset

# import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.filters import gaussian_filter

# import logging


def build_transforms(opts, train=True):
    tr = []
    if train:
        pre_transform = T.Compose([T.FaceToEdge(), T.Constant(value=1)])

    # --- transform data(ndarrays) to tensor
    tr.append(toTensor())
    # --- Normalize to 0-mean, 1-var
    # TODO borrar?
    # tr.append(normalizeTensor(mean=None, std=None))

    # --- add all trans to que tranf queue
    return tv_transforms.Compose(tr)


class toTensor(object):
    """Convert dataset sample (ndarray) to tensor"""

    def __call__(self, sample):
        # for k, v in sample.iteritems():
        for k, v in list(sample.items()):
            if type(v) is np.ndarray:
                # --- by default float arrays will be converted to float tensors
                # --- and int arrays to long tensor.
                sample[k] = torch.from_numpy(v)
        return sample
