#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 16:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   config.py
# @Desc     :   

from dataclasses import dataclass, field
from pathlib import Path
from torch import cuda

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class FilePaths:
    MODEL: Path = BASE_DIR / "params/model.pth"
    DATASET_TRAIN = BASE_DIR / "data/train/"
    DATASET_TEST = BASE_DIR / "data/test/"


@dataclass
class DataPreprocessor:
    PCA_VARIANCE_THRESHOLD: float = 0.95
    RANDOM_STATE: int = 27
    VALID_SIZE: float = 0.2
    IS_SHUFFLE: bool = True
    BATCHES: int = 16


@dataclass
class RNNParams:
    DROPOUT_RATE: float = 0.5
    RNN_EMBEDDING_DIM: int = 256
    RNN_HIDDEN_SIZE: int = 128
    RNN_LAYERS: int = 2
    RNN_TEMPERATURE: float = 1.0


@dataclass
class UNetParams:
    SEG_CLASSES: int = 32


@dataclass
class Hyperparameters:
    ALPHA: float = 1e-3
    EPOCHS: int = 50
    ACCELERATOR: str = "cuda" if cuda.is_available() else "cpu"


@dataclass
class Configration:
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)
    RNN_PARAMS: RNNParams = field(default_factory=RNNParams)
    UNET_PARAMS: UNetParams = field(default_factory=UNetParams)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)


CONFIG = Configration()
