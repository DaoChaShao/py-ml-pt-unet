#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 16:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :   

from numpy import array, sum as np_sum
from pathlib import Path
from PIL import Image
from pandas import DataFrame, read_csv
from random import randint
from torch import optim, nn
from torchvision import transforms

from dataseter.unet_seg_setter import MaskToClassID, UNetDataset
from models.unet_seg_model import UNetDoubleConvModel
from trainers.train_unet_seg import UNetSegmentationTrainer
from utils.config import CONFIG
from utils.helper import load_data_paths, RandomSeed
from utils.highlighter import red, green
from utils.PT import TorchDataLoader


def verify_classes_convention(class2id: dict, mask: Image.Image) -> None:
    """ Check the classes in the mask """
    mask = array(mask)

    # Calculate unique classes in the mask
    mask_classes = set(mask.flatten().tolist())
    unique_classes = set(class2id.values())
    mapped_classes = mask_classes.intersection(unique_classes)
    print(f"There are {len(unique_classes)} classes.")
    print(f"There are {green(len(mapped_classes))} classes in the mask found in the class2id mapping.")
    print(f"There are {red(len(unique_classes - mapped_classes))} classes in the mask not mapped in class2id.")


def preprocess_data():
    """ Preprocess the data """
    base_train = Path(CONFIG.FILEPATHS.DATASET_TRAIN)
    base_valid = Path(CONFIG.FILEPATHS.DATASET_VALID)
    # print(base_train)
    # print(base_valid)

    paths_train: tuple[list[str], list[str]] = load_data_paths(base_train)
    paths_valid: tuple[list[str], list[str]] = load_data_paths(base_valid)
    # print(len(paths_train[0]), len(paths_valid[0]))
    # print(paths_train[0][:3])
    # print(paths_train[1][:3])
    # print(paths_valid[0][:3])
    # print(paths_valid[1][:3])

    # index_train = randint(0, len(paths_train[0]) - 1)
    # img_train = Image.open(paths_train[0][index_train])
    # mask_train = Image.open(paths_train[1][index_train])
    # print(f"Random train index selected: {index_train}")
    # print(f"The image info in train dataset is {img_train.size} {img_train.mode}")
    # print(f"The mask info in train dataset is {mask_train.size} {mask_train.mode}")

    # index_valid = randint(0, len(paths_valid[0]) - 1)
    # img_valid = Image.open(paths_valid[0][index_valid])
    # mask_valid = Image.open(paths_valid[1][index_valid])
    # print(f"Random valid index selected: {index_valid}")
    # print(f"The image info in test dataset is {img_valid.size} {img_valid.mode}")
    # print(f"The mask info in test dataset is {mask_valid.size} {img_valid.mode}")

    classes: DataFrame = read_csv(CONFIG.FILEPATHS.CLASSES_DICTIONARY)
    # print(f"The number of segmentation classes: {len(classes)}")
    # print(f"Segmentation classes dictionary:\n{classes}")
    class2id: dict[tuple[int, int, int], int] = {}
    for i in range(len(classes)):
        row = classes.iloc[i]
        class2id[(
            int(row["r"]),
            int(row["g"]),
            int(row["b"])
        )] = i
    # print(f"Total number of classes: {len(class2id)}")
    # print(f"Class to ID mapping:\n{class2id}")

    return paths_train, paths_valid, class2id


def prepare_dataset():
    # preprocess_data()
    paths_train, paths_valid, class2id = preprocess_data()

    # Check the mask metric
    # index_check: int = randint(0, len(paths_train[1]) - 1)
    # print(paths_train[1][index_check])
    # mask_check: Image.Image = Image.open(paths_train[1][index_check]).convert("RGB")
    # converter = MaskToClassID(class2id)
    # mask_converted: Image.Image = converter(mask_check)
    # print(f"Random index selected for mask check: {index_check}")
    # print(f"Original mask size: {mask_check.size}, mode: {mask_check.mode}")
    # print(f"Converted mask size: {mask_converted.size}, mode: {mask_converted.mode}")
    # print(array(mask_check))
    # print(array(mask_converted))
    # verify_classes_convention(class2id, mask_converted)

    # Setup image enhancements
    img_transformer = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    mask_transformer = transforms.Compose([
        transforms.Resize((512, 512)),
    ])

    # Setup datasets
    dataset_train = UNetDataset(
        image_paths=paths_train[0],
        mask_paths=paths_train[1],
        class2id=class2id,
        img_transformer=img_transformer,
        mask_transformer=mask_transformer
    )
    dataset_valid = UNetDataset(
        image_paths=paths_valid[0],
        mask_paths=paths_valid[1],
        class2id=class2id,
        img_transformer=img_transformer,
        mask_transformer=mask_transformer
    )
    # index_train: int = randint(0, len(dataset_train) - 1)
    # index_valid: int = randint(0, len(dataset_valid) - 1)
    # img_train, mask_train = dataset_train[index_train]
    # img_valid, mask_valid = dataset_valid[index_valid]
    # print(f"Transformed train image shape: {img_train.shape}, mask shape: {mask_train.shape}")
    # print(f"Transformed test image shape: {img_valid.shape}, mask shape: {mask_valid.shape}")
    # print(mask_train.numpy(), mask_valid.numpy(), sep="\n")

    # Set up dataloader
    dataloader_train = TorchDataLoader(
        dataset=dataset_train,
        batch_size=CONFIG.PREPROCESSOR.BATCHES,
        is_shuffle=CONFIG.PREPROCESSOR.IS_SHUFFLE,
    )
    dataloader_valid = TorchDataLoader(
        dataset=dataset_valid,
        batch_size=CONFIG.PREPROCESSOR.BATCHES,
        is_shuffle=CONFIG.PREPROCESSOR.IS_SHUFFLE,
    )

    # print(f"Number of training batches: {len(dataloader_train)}")
    # print(f"Number of validation batches: {len(dataloader_valid)}")

    return dataloader_train, dataloader_valid, class2id


def main() -> None:
    """ Main Function """
    with RandomSeed("Training UNet Segmentation Model"):
        # prepare_dataset()
        train, valid, class2id = prepare_dataset()

        # Set up a model
        channels, height, width = train[0][0].shape
        # print(channels, height, width)
        model = UNetDoubleConvModel(channels, CONFIG.UNET_PARAMS.SEG_CLASSES, height, width)

        # Set up an optimiser and loss function
        optimiser = optim.AdamW(model.parameters(), lr=CONFIG.HYPERPARAMETERS.ALPHA)
        criterion = nn.CrossEntropyLoss()

        # Initialise a trainer
        trainer = UNetSegmentationTrainer(
            model, optimiser, criterion,
            len(class2id),
            CONFIG.HYPERPARAMETERS.ACCELERATOR
        )
        trainer.fit(train, valid, CONFIG.HYPERPARAMETERS.EPOCHS, CONFIG.FILEPATHS.MODEL)


if __name__ == "__main__":
    main()
