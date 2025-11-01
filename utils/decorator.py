#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 16:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   decorator.py
# @Desc     :   

from time import perf_counter
from functools import wraps


def timer(func):
    """ The decorator for timing functions
    :param func: The function to be decorated
    :return: The decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print("*" * 50)
        print(f"The function named {func.__name__!r} is starting:")
        print("-" * 50)
        time_start = perf_counter()
        result = func(*args, **kwargs)
        time_end = perf_counter()
        time_elapsed = time_end - time_start
        print("-" * 50)
        print(f"The function named {func.__name__!r} took {time_elapsed:.4f} seconds to complete.")
        print("*" * 50)
        print()
        return result

    return wrapper


def beautifier(func):
    """ The decorator for beautifying function output
    :param func: The function to be decorated
    :return: The decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print("*" * 50)
        print(f"The function named {func.__name__!r} is starting:")
        print("-" * 50)
        result = func(*args, **kwargs)
        print("-" * 50)
        print(f"The function named {func.__name__!r} has completed.")
        print("*" * 50)
        print()
        return result

    return wrapper


class Timer(object):
    """ Class timer decorator with precision control """

    def __init__(self, description: str, precision: int = 6):
        self._desc = description
        self._precision = precision

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            print("*" * 50)
            print(f"Function {func.__name__} is starting:")
            print("-" * 50)
            _start = perf_counter()
            result = func(*args, **kwargs)
            _end = perf_counter()
            print(result)
            print("-" * 50)
            print(f"Function {func.__name__} has ended.")
            print("-" * 50)
            print(f"{self._desc} took {(_end - _start):.{self._precision}f} seconds.")
            print("*" * 50)
            return result

        return wrapper
