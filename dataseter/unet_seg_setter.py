#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 21:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   unet_seg_model.py
# @Desc     :   

from numpy import array
from PIL import Image
from torch import Tensor, from_numpy
from torch.utils.data import Dataset


class MaskToClassID:
    """ A callable class to convert mask pixel values to class IDs """

    def __init__(self, class2id: dict):
        """ Initialise the MashToClassID class """
        self._class2id = class2id

    def __call__(self, mask: Image.Image) -> Image.Image:
        """ Convert mask pixel values to class IDs """
        width, height = mask.size
        mask2id = Image.new("L", (width, height))
        for w in range(width):
            for h in range(height):
                pixel = mask.getpixel((w, h))
                class2id = self._class2id.get(pixel, 0)
                mask2id.putpixel((w, h), class2id)

        return mask2id


class UNetDataset(Dataset):
    """ A custom PyTorch Dataset class for UNet model """

    def __init__(self,
                 image_paths: list[str], mask_paths: list[str],
                 class2id: dict,
                 img_transformer=None, mask_transformer=None
                 ):
        """ Initialise the UNetDataset class """
        self._image_paths = image_paths
        self._mask_paths = mask_paths
        self._converter = MaskToClassID(class2id)
        self._img_transformer = img_transformer
        self._mask_transformer = mask_transformer

    def __len__(self) -> int:
        """ Return the total number of samples in the dataset """
        return len(self._image_paths)

    def __getitem__(self, index: int) -> tuple:
        """ Return a single (feature, label) pair or a batch via slice """
        image: Image.Image = Image.open(self._image_paths[index]).convert("RGB")
        if self._img_transformer:
            image = self._img_transformer(image)

        mask: Image.Image = Image.open(self._mask_paths[index]).convert("RGB")
        if self._mask_transformer:
            mask = self._mask_transformer(mask)

        converted_mask: Image.Image = self._converter(mask)
        mask: Tensor = from_numpy(array(converted_mask)).long()

        return image, mask


if __name__ == "__main__":
    pass
