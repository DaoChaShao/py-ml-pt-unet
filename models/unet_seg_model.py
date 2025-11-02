#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 22:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   unet_seg_model.py
# @Desc     :   

from torch import nn, cat
from torchsummary import summary


class InputDoubleConv(nn.Module):
    """ A double convolutional layer block for UNet input processing """

    def __init__(self, in_channels: int, out_channels: int, mid_channels=None, height=None, width=None):
        super().__init__()
        self._in = in_channels
        self._H = height
        self._W = width
        if not mid_channels:
            mid_channels = out_channels

        # Setup input layers using nn.Sequential
        self._input = nn.Sequential(
            nn.Conv2d(self._in, mid_channels, kernel_size=3, padding=1, bias=True),  # Keep original size
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),  # Keep original size
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Initialise parameters
        self._input.apply(self._initialise_params)

    @staticmethod
    def _initialise_params(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self._input(x)

    def summary(self):
        # input size: (batch, channels, height, width)
        if self._H is None or self._W is None:
            raise ValueError("Height and Width must be specified for model summary.")

        summary(self, input_size=(1, self._in, self._H, self._W))
        print(f"Model Summary for {self.__class__.__name__}")
        print("=" * 64)
        print(self)
        print("=" * 64)
        print()


class DownSampler(nn.Module):
    """ Encoding down-sampling block for UNet model """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._pool = nn.Sequential(
            nn.MaxPool2d(2),
            InputDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self._pool(x)

    def summary(self):
        print("=" * 64)
        print(self)
        print("=" * 64)
        print()


class UpSampler(nn.Module):
    """ Decoding up-sampling block for UNet model """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self._up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self._conv = InputDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self._up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self._conv = InputDoubleConv(in_channels, out_channels)

    def forward(self, decoder_feat, encoder_feat):
        # Decode the features
        decoder_out = self._up(decoder_feat)

        # Find the size difference and pad if necessary, size: (batch, channels, height, width)
        diff_H = encoder_feat.size()[2] - decoder_out.size()[2]
        diff_W = encoder_feat.size()[3] - decoder_out.size()[3]

        # Pad the decoder output to match encoder feature size: [left, right, top, bottom]
        padded = nn.functional.pad(decoder_out, [diff_W // 2, diff_W - diff_W // 2, diff_H // 2, diff_H - diff_H // 2])
        # Fusing encoder and decoder features - channel-wise concatenation (Skip Connections)
        out = cat([encoder_feat, padded], dim=1)

        # Apply the convolutional layers
        return self._conv(out)

    def summary(self):
        print("=" * 64)
        print(self)
        print("=" * 64)
        print()


class OutputConv(nn.Module):
    """ Output convolutional layer for UNet model """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._output = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self._output(x)

    def summary(self):
        print("=" * 64)
        print(self)
        print("=" * 64)
        print()


class UNetDoubleConvModel(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, height: int, width: int, bilinear: bool = True):
        super().__init__()
        self._in = in_channels
        self._classes = n_classes
        self._H = height
        self._W = width
        self._B = bilinear

        self._inc = InputDoubleConv(self._in, 64, 64, height=self._H, width=self._W)
        self._down_i = DownSampler(64, 128)
        self._down_ii = DownSampler(128, 256)
        self._down_iii = DownSampler(256, 512)

        factor = 2 if self._B else 1
        self._down_iv = DownSampler(512, 1024 // factor)

        self._up_i = UpSampler(1024, 512 // factor, bilinear=self._B)
        self._up_ii = UpSampler(512, 256 // factor, bilinear=self._B)
        self._up_iii = UpSampler(256, 128 // factor, bilinear=self._B)
        self._up_iv = UpSampler(128, 64, bilinear=self._B)

        self._outc = OutputConv(64, self._classes)

        self._initialise_params()

    def _initialise_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        out = self._inc(x)

        down_i = self._down_i(out)
        down_ii = self._down_ii(down_i)
        down_iii = self._down_iii(down_ii)
        down_iv = self._down_iv(down_iii)

        up_i = self._up_i(down_iv, down_iii)
        up_ii = self._up_ii(up_i, down_ii)
        up_iii = self._up_iii(up_ii, down_i)
        up_iv = self._up_iv(up_iii, out)

        logits = self._outc(up_iv)

        return logits

    def summary(self):
        # input size: (batch, channels, height, width)
        summary(self, input_size=(1, self._in, self._H, self._W))
        print(f"Model Summary for {self.__class__.__name__}")
        print("=" * 64)
        print(self)
        print("=" * 64)
        print()


if __name__ == "__main__":
    pass
