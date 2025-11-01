#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 21:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   segment.py
# @Desc     :   

from PIL import Image
from random import randint
from torch.utils.data import Dataset


class UNetDataset(Dataset):
    """ A custom PyTorch Dataset class for UNet model """

    def __init__(self, image_paths: list[str], mask_paths: list[str], img_transformer=None, mask_transformer=None):
        """ Initialise the UNetDataset class """
        self._image_paths = image_paths
        self._mask_paths = mask_paths
        self._img_transformer = img_transformer
        self._mask_transformer = mask_transformer
        self._index = randint(0, len(self._image_paths) - 1)

        if len(self._image_paths) > 0:
            img = Image.open(self._image_paths[self._index])
            mask = Image.open(self._mask_paths[self._index])
            assert img.size == mask.size, f"Image and mask shapes do not match: {img.size} vs {mask.size}"

    def __len__(self, ) -> int:
        """ Return the total number of samples in the dataset """
        return len(self._image_paths)

    def __getitem__(self, index: int) -> tuple:
        """ Return a single (feature, label) pair or a batch via slice """
        image = Image.open(self._image_paths[index]).convert("RGB")
        mask = Image.open(self._mask_paths[index]).convert("L")

        if self._img_transformer:
            image = self._img_transformer(image)
        if self._mask_transformer:
            mask = self._mask_transformer(mask)

        return image, mask


if __name__ == "__main__":
    pass
