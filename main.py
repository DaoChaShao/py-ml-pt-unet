#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 16:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :   

from pathlib import Path
from PIL import Image
from random import randint
from torch import optim, nn
from torchvision import transforms

from dataseter.unet import UNetDataset
from models.segment import UNetDoubleConvModel
from trainers.unet_seg import UNetSegmentationTrainer
from utils.config import CONFIG
from utils.helper import load_data_paths
from utils.PT import TorchDataLoader


def preprocess_data():
    """ Preprocess the data """
    path_train = Path(CONFIG.FILEPATHS.DATASET_TRAIN)
    path_valid = Path(CONFIG.FILEPATHS.DATASET_TEST)
    # print(path_train)
    # print(path_valid)

    train: tuple[list[str], list[str]] = load_data_paths(path_train)
    valid: tuple[list[str], list[str]] = load_data_paths(path_valid)
    # print(raw_train[0][:3])
    # print(raw_train[1][:3])

    index = randint(0, len(train) - 1)
    img_train = Image.open(train[0][index])
    mask_train = Image.open(train[1][index])
    print(f"The image info in train dataset is {img_train.size} {img_train.mode}")
    print(f"The mask info in train dataset is {mask_train.size} {mask_train.mode}")

    img_valid = Image.open(valid[0][index])
    mask_valid = Image.open(valid[1][index])
    print(f"The image info in valid dataset is {img_valid.size} {img_valid.mode}")
    print(f"The mask info in valid dataset is {mask_valid.size} {img_valid.mode}")

    return train, valid


def prepare_dataset():
    train, valid = preprocess_data()
    print(len(train[0]), len(valid[0]))

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
        transforms.ToTensor(),
    ])

    # Setup datasets
    dataset_train = UNetDataset(
        image_paths=train[0],
        mask_paths=train[1],
        img_transformer=img_transformer,
        mask_transformer=mask_transformer
    )
    dataset_valid = UNetDataset(
        image_paths=valid[0],
        mask_paths=valid[1],
        img_transformer=img_transformer,
        mask_transformer=mask_transformer
    )
    index_train: int = randint(0, len(dataset_train) - 1)
    index_valid: int = randint(0, len(dataset_valid) - 1)
    img_train, mask_train = dataset_train[index_train]
    img_valid, mask_valid = dataset_valid[index_valid]
    # print(index_train, index_valid)
    # print(img_train, img_valid)
    print(f"Transformed train image shape: {img_train.shape}, mask shape: {mask_train.shape}")
    print(f"Transformed valid image shape: {img_valid.shape}, mask shape: {mask_valid.shape}")

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

    return dataloader_train, dataloader_valid


def main() -> None:
    """ Main Function """
    train, valid = prepare_dataset()

    # Set up a model
    channels, height, width = train[0][0].shape
    print(channels, height, width)
    model = UNetDoubleConvModel(channels, CONFIG.UNET_PARAMS.SEG_CLASSES, height, width, )

    # Set up an optimiser and loss function
    optimiser = optim.AdamW(model.parameters(), lr=CONFIG.HYPERPARAMETERS.ALPHA)
    criterion = nn.CrossEntropyLoss()

    # Initialise a trainer
    trainer = UNetSegmentationTrainer(model, optimiser, criterion, CONFIG.HYPERPARAMETERS.ACCELERATOR)
    trainer.fit(train, valid, CONFIG.HYPERPARAMETERS.EPOCHS, CONFIG.FILEPATHS.MODEL)


if __name__ == "__main__":
    main()
