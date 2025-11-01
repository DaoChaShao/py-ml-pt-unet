#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 21:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   linear.py
# @Desc     :   

from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor, tensor, float32, int64
from typing import Any, Union

from torch.utils.data import Dataset


class LabelTorchDataset(Dataset):
    """ A custom PyTorch Dataset class for handling label features and masks """

    def __init__(self, features: Any, labels: Any):
        """ Initialise the TorchDataset class
        :param features: the feature tensor
        :param labels: the label tensor
        """
        self._features: Tensor = self._to_tensor(features)
        self._labels: Tensor = self._to_tensor(labels)

    @property
    def features(self) -> Tensor:
        """ Return the feature tensor as a property """
        return self._features

    @property
    def labels(self) -> Tensor:
        """ Return the label tensor as a property """
        return self._labels

    @staticmethod
    def _to_tensor(data: Union[DataFrame, Tensor, ndarray, list], is_label: bool = False) -> Tensor:
        """ Convert input data to a PyTorch tensor on the specified device
        :param data: the input data (DataFrame, ndarray, list, or Tensor)
        :param is_label: whether the data is label data
        :return: the converted PyTorch tensor
        """
        if isinstance(data, (DataFrame, Series)):
            out = tensor(data.values, dtype=float32 if not is_label else int64)
        elif isinstance(data, Tensor):
            out = data.float() if not is_label else data.long()
        elif isinstance(data, (ndarray, list)):
            out = tensor(data, dtype=float32 if not is_label else int64)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return out

    def __len__(self) -> int:
        """ Return the total number of samples in the dataset """
        return len(self._features)

    def __getitem__(self, index: Union[int, slice]) -> Union[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """ Return a single (feature, label) pair or a batch via slice """
        if isinstance(index, slice):
            # Return a batch (for example dataset[:5])
            return self._features[index], self._labels[index]
        elif isinstance(index, int):
            # Return a single sample
            return self._features[index], self._labels[index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __repr__(self):
        """ Return a string representation of the dataset """
        return f"LabelTorchDataset(features={self._features.shape}, masks={self._labels.shape}, device=cpu)"
