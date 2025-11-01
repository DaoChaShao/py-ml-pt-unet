#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 16:18
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   criterion.py
# @Desc     :   

from torch import Tensor, nn, clamp, log, sqrt


def log_mse_loss(pred: Tensor, true: Tensor, epsilon: float = 1e-7) -> Tensor:
    """ Calculate the Log Mean Squared Error Loss between predictions and true values
    - This loss is useful for regression tasks where the target values span several orders of magnitude (e.g., housing prices, population counts)
    - This loss should NOT be used when the data contains negative values
    - Predictions and targets must be strictly positive
    - The loss computes MSE in log-space, which emphasizes relative errors rather than absolute errors
    :param pred: the predicted tensor
    :param true: the true tensor
    :param epsilon: small value to avoid log(0)
    :return: the Log MSE Loss tensor
    """
    criterion = nn.MSELoss()

    pred_clamped = clamp(pred, 1, float("inf"))
    true_clamped = clamp(true, 1, float("inf"))

    pred_log = log(pred_clamped + epsilon)
    true_log = log(true_clamped + epsilon)

    return sqrt(criterion(pred_log, true_log))
