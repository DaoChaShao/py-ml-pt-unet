#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/2 11:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from pathlib import Path

import torch
from PIL import Image
from random import randint
from torch import load
from torchvision import transforms

from models.unet_seg_model import UNetDoubleConvModel
from utils.config import CONFIG
from utils.helper import load_data_paths


def main() -> None:
    """ Main Function """
    path = Path(CONFIG.FILEPATHS.MODEL)

    if path.exists():
        print(f"Model {path.name} has been trained and saved.")

        base = Path(CONFIG.FILEPATHS.DATASET_VALID)
        paths = load_data_paths(base)

        index = randint(0, len(paths[0]) - 1)
        img_path = paths[0][index]
        mask_path = paths[1][index]
        print(f"Selected index: {index}")
        print(img_path)
        print(mask_path)

        # Conduct the image
        img_transformer = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        pil = Image.open(img_path).convert("RGB")
        # mask = Image.open(mask_path).convert("L")
        img = img_transformer(pil).unsqueeze(0)
        print(img.shape)

        # Load the model
        _, channels, height, width = img.shape
        print(channels, height, width)
        model = UNetDoubleConvModel(channels, CONFIG.UNET_PARAMS.SEG_CLASSES, height=height, width=width)
        state_dict = load(path)
        model.load_state_dict(state_dict)
        print("Model state keys:", list(state_dict.keys())[:3])
        print("Model parameters:")
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                print(f"{name}: {param.shape} - mean: {param.mean():.3f}")
                break
        model.eval()

        with torch.no_grad():
            output = model(img)
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            # 检查每个通道的最大值
            channel_max = output.squeeze(0).amax(dim=(1, 2))  # 每个通道的最大值
            print(f"Each channel max values: {channel_max}")

            # 检查哪个通道的得分最高
            max_channel = channel_max.argmax().item()
            print(f"Channel with highest score: {max_channel}")

            # 检查softmax概率
            softmax_probs = torch.softmax(output, dim=1)
            class_probs = softmax_probs.squeeze(0).mean(dim=(1, 2))  # 每个类别的平均概率
            print(f"Average probability per class: {class_probs}")

            # 检查预测的置信度
            confidence = softmax_probs.max(dim=1)[0].mean().item()
            print(f"Average prediction confidence: {confidence:.3f}")

            prediction = output.argmax(dim=1).squeeze()
            print(f"Prediction shape: {prediction.shape}")
            print(f"Prediction unique values: {torch.unique(prediction)}")
    else:
        print(f"Model {path.name} path not found: {path}")


if __name__ == "__main__":
    main()
