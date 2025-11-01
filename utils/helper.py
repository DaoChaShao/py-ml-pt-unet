#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 16:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :   

from dataclasses import dataclass
from json import load, dump
from random import seed as rnd_seed, getstate, setstate
from pathlib import Path
from pandas import DataFrame, read_csv
from time import perf_counter
from tqdm import tqdm

from utils.decorator import timer

LENGTH: int = 50


class Timer(object):
    """ timing code blocks using a context manager """

    def __init__(self, description: str = None, precision: int = 5):
        """ Initialise the Timer class
        :param description: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        self._description: str = description
        self._precision: int = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Start the timer """
        self._start = perf_counter()
        print("*" * LENGTH)
        print(f"{self._description} has started.")
        print("-" * LENGTH)
        return self

    def __exit__(self, *args):
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

        print("-" * LENGTH)
        print(f"{self._description} took {self._elapsed:.{self._precision}f} seconds.")
        print("*" * LENGTH)

    def __repr__(self):
        """ Return a string representation of the timer """
        if self._elapsed != 0.0:
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
        return f"{self._description} has NOT started."


class Beautifier(object):
    """ beautifying code blocks using a context manager """

    def __init__(self, description: str = None):
        """ Initialise the Beautifier class
        :param description: the description of a beautifier
        """
        self._description: str = description

    def __enter__(self):
        """ Start the beautifier """
        print("*" * LENGTH)
        print(f"The block named {self._description!r} is starting:")
        print("-" * LENGTH)
        return self

    def __exit__(self, *args):
        """ Stop the beautifier """
        print("-" * LENGTH)
        print(f"The block named {self._description!r} has completed.")
        print("*" * LENGTH)
        print()


class RandomSeed:
    """ Setting random seed for reproducibility """

    def __init__(self, description: str, seed: int = 27):
        """ Initialise the RandomSeed class
        :param description: the description of a random seed
        :param seed: the seed value to be set
        """
        self._description: str = description
        self._seed: int = seed
        self._previous_py_seed = None

    def __enter__(self):
        """ Set the random seed """
        # Save the previous random seed state
        self._previous_py_seed = getstate()

        # Set the new random seed
        rnd_seed(self._seed)

        print("*" * LENGTH)
        print(f"{self._description!r} has been set randomness {self._seed}.")
        print("-" * LENGTH)

        return self

    def __exit__(self, *args):
        """ Exit the random seed context manager """
        # Restore the previous random seed state
        if self._previous_py_seed is not None:
            setstate(self._previous_py_seed)

        print("-" * LENGTH)
        print(f"{self._description!r} has been restored to previous randomness.")
        print("*" * LENGTH)
        print()

    def __repr__(self):
        """ Return a string representation of the random seed """
        return f"{self._description!r} is set to randomness {self._seed}."


@dataclass
class Outputter:
    """ Print Class with ON/OFF switch """
    _enabled: bool = True

    def yes(self):
        """ Open output """
        self._enabled = True

    def no(self):
        """ Close output """
        self._enabled = False

    def print(self, msg: str) -> None:
        """ Display the message if switch is on """
        if self._enabled:
            print(msg)


out = Outputter()


@timer
def read_file(file_path: str | Path) -> str:
    """ Read content from a file
    :param file_path: path to the file
    :return: content read from the file
    """
    with open(str(file_path), "r", encoding="utf-8") as file:
        content = file.read()
        print(f"The content:\n{content}")
    return content


@timer
def load_text_data(text_data_path: str, cols: bool = False, columns: list | None = None) -> DataFrame:
    """ Read data from a txt file with a structural data format
    :param text_data_path: path to the text data file
    :param cols: whether to specify column names
    :param columns: list of column names
    :return: data read from the text file
    """
    if cols:
        data: DataFrame = read_csv(text_data_path, names=columns, sep=r"\s+")
    else:
        data: DataFrame = read_csv(text_data_path, sep=r"\s+")

    print(f"Loaded text data' shape is {data.shape}.")

    return data


@timer
def load_json(json_path: str | Path) -> dict:
    """ Load JSON data from a file
    :param json_path: path to the JSON file
    :return: data loaded from the JSON file
    """
    with open(str(json_path), "r", encoding="utf-8") as file:
        data: dict = load(file)

    print(f"JSON data loaded from {json_path}:")

    return data


@timer
def save_json(json_data: dict, json_path: str | Path) -> None:
    with open(str(json_path), "w", encoding="utf-8") as file:
        dump(json_data, file, indent=2)


@timer
def load_text_data(filepath: str | Path):
    """ Load text data from a directory
    :param filepath: path to the directory
    :return: concatenated text data from all files in the directory
    """
    data: dict[str, list] = {
        "ids": [],
        "contents": [],
        "ratings": [],
        "masks": [],
    }
    base: Path = Path(filepath)
    if base.exists():
        for subdir in base.iterdir():
            if subdir.name == "pos":
                label: int = 1
            else:
                label: int = 0
            if subdir.is_dir():
                for file in tqdm(subdir.iterdir(), total=12500, desc=f"Processing {subdir.name} in {base.name}"):
                    if file.suffix == ".txt":
                        with open(str(file), "r", encoding="utf-8") as f:
                            content: str = f.read().strip()
                        names: list[str] = file.stem.split("_")
                        data["ids"].append(int(names[0]))
                        data["contents"].append(str(content))
                        data["ratings"].append(int(names[1]))
                        data["masks"].append(int(label))

        print(f"Loaded {len(data["ids"])} text files from directory: {filepath}")
    else:
        print(f"The directory {filepath} does not exist.")

    return data


@timer
def load_data_paths(dir_path: Path) ->  tuple[list[str], list[str]]:
    """ Load all file paths from a directory
    :param dir_path: path to the directory
    :return: list of file paths in the directory
    """
    images: list = []
    labels: list = []

    if dir_path.exists() and dir_path.is_dir():
        for subdir in dir_path.iterdir():
            match subdir.name:
                case "images":
                    for file in subdir.iterdir():
                        if file.is_file():
                            images.append(file.resolve())
                case "masks":
                    for file in subdir.iterdir():
                        if file.is_file():
                            labels.append(file.resolve())

        images.sort(key=lambda x: x.name)
        labels.sort(key=lambda x: x.name)

        print(f"Loaded {len(images)} files from directory: {dir_path.name}")
    else:
        print(f"The directory {dir_path} does not exist or is not a directory.")

    return images, labels
