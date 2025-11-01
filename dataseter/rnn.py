#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 21:15
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   rnn.py
# @Desc     :   

from torch import Tensor, tensor, long
from torch.utils.data import Dataset
from typing import Union


class SeqPredictionTorchDataset(Dataset):
    """ A custom PyTorch Dataset class for handling sequential features and masks """

    def __init__(self, sequences: list, seq_max_len: int, pad_token: int) -> None:
        """ Initialise the TorchDataset class for sequential data
        :param sequences: the input sequences
        :param seq_max_len: the length of each sequence
        :param pad_token: the padding token to use
        """
        self._sequences = sequences
        self._length = seq_max_len
        self._pad = pad_token
        self._features, self._labels = self._pad_to_seq2seq_tensor()

    def _pad_to_seq2one_tensor(self) -> tuple[Tensor, Tensor]:
        """ Convert input data to a PyTorch tensor via padding for one-step prediction
        :return: the converted PyTorch tensor
        """
        _features, _labels = [], []
        for i in range(len(self._sequences) - 1):
            if i < self._length - 1:
                feature = [self._pad] * (self._length - i - 1) + self._sequences[0: i + 1]
            else:
                feature = self._sequences[i - self._length + 1: i + 1]
            label = self._sequences[i + 1]
            _features.append(feature)
            _labels.append(label)

        return tensor(_features), tensor(_labels)

    def _pad_to_seq2seq_tensor(self) -> tuple[Tensor, Tensor]:
        """ Convert input data to a PyTorch tensor via sequence padding for sequence-to-sequence prediction
        :return: the converted PyTorch tensor
        """
        _features, _labels = [], []
        for i in range(len(self._sequences) - 1):
            if i < self._length - 1:
                feature = [self._pad] * (self._length - i - 1) + self._sequences[0: i + 1]
                label = [self._pad] * (self._length - i - 1) + self._sequences[1: i + 2]
            else:
                feature = self._sequences[i - self._length + 1: i + 1]
                label = self._sequences[i - self._length + 2: i + 2]

            _features.append(feature)
            _labels.append(label)

        return tensor(_features), tensor(_labels)

    def _slice_to_tensor(self) -> tuple[Tensor, Tensor]:
        """ Convert input data to a PyTorch tensor via sliding window for next-step prediction
        :return: the converted PyTorch tensor
        """
        _features, _labels = [], []
        for i in range(len(self._sequences) - self._length):
            feature = self._sequences[i: i + self._length]
            label = self._sequences[i + self._length]
            _features.append(feature)
            _labels.append(label)

        return tensor(_features), tensor(_labels)

    @property
    def features(self) -> Tensor:
        """ Return the feature tensor as a property """
        return self._features

    @property
    def labels(self) -> Tensor:
        """ Return the label tensor as a property """
        return self._labels

    def __len__(self) -> int:
        """ Return the total number of samples in the dataset """
        return len(self._features)

    def __getitem__(self, index: Union[int, slice]) -> tuple[Tensor, Tensor]:
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
        return f"SequentialTorchDataset(features={self._features.shape}, masks={self._labels.shape}, device={self._features.device})"


class SeqClassificationTorchDataset(Dataset):
    """ A custom PyTorch Dataset class for handling sequential features and masks """

    def __init__(self, feature_seqs: list, lbl_seqs: list, seq_max_len: int, pad_token: int = 0) -> None:
        """ Initialise the TorchDataset class for sequential data
        :param feature_seqs: the input sequences
        :param lbl_seqs: the label sequences
        :param seq_max_len: the length of each sequence
        :param pad_token: the padding token to use
        """
        self._sequences = feature_seqs
        self._length = seq_max_len
        self._pad = pad_token
        self._features = self._pad_to_fixed_len_tensor()
        self._labels = tensor(lbl_seqs, dtype=long)

    def _pad_to_fixed_len_tensor(self) -> Tensor:
        """ Convert input data to a PyTorch tensor via padding to fixed length
        :return: the converted PyTorch tensor
        """
        _features = []
        for seq in self._sequences:
            if len(seq) < self._length:
                padded_seq = seq + [self._pad] * (self._length - len(seq))
            else:
                padded_seq = seq[:self._length]
            _features.append(padded_seq)

        return tensor(_features)

    @property
    def features(self) -> Tensor:
        """ Return the feature tensor as a property """
        return self._features

    @property
    def labels(self) -> Tensor:
        """ Return the label tensor as a property """
        return self._labels

    def __len__(self) -> int:
        """ Return the total number of samples in the dataset """
        return len(self._features)

    def __getitem__(self, index: Union[int, slice]) -> tuple[Tensor, Tensor]:
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
        return f"SequentialTorchDataset(features={self._features.shape}, masks={self._labels.shape}, device={self._features.device})"
