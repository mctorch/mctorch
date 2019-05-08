"""
This module converts objects into numpy array.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import six

from caffe2.python import workspace


def make_np(x):
    """
    Args:
      x: An instance of torch tensor or caffe blob name

    Returns:
        numpy.array: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, six.string_types):  # Caffe2 will pass name of blob(s) to fetch
        return _prepare_caffe2(x)
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, torch.Tensor):
        return _prepare_pytorch(x)
    raise NotImplementedError(
        'Got {}, but numpy array, torch tensor, or caffe2 blob name are expected.'.format(type(x)))


def _prepare_pytorch(x):
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    x = x.cpu().numpy()
    return x


def _prepare_caffe2(x):
    x = workspace.FetchBlob(x)
    return x
