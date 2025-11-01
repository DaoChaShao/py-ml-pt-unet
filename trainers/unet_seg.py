#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/1 14:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   unet_seg.py
# @Desc     :   

from PySide6.QtCore import QObject, Signal
from torch import nn, no_grad, save, device, Tensor

from utils.PT import get_device, DataLoader, TorchDataLoader


class UNetSegmentationTrainer(QObject):
    """ Trainer class for managing training process """
    losses: Signal = Signal(int, float, float, float)

    def __init__(self, model: nn.Module, optimiser, criterion, accelerator: str = "auto") -> None:
        super().__init__()
        self._model = model
        self._optimiser = optimiser
        self._criterion = criterion
        self._accelerator = get_device(accelerator)

    def _epoch_train(self, dataloader: DataLoader | TorchDataLoader) -> float:
        """ Train the model for one epoch
        :param dataloader: DataLoader for training data
        :return: average training loss for the epoch
        """
        # Set model to training mode
        self._model.train()

        _loss: float = 0.0
        _total: float = 0.0
        for features, labels in dataloader:
            features, labels = features.to(device(self._accelerator)), labels.to(device(self._accelerator))

            self._optimiser.zero_grad()
            outputs = self._model(features)
            # print(outputs.shape, masks.shape)

            # Transform original size (B, 1, H, W) into new size (B, H, W) because of using CrossEntropyLoss
            labels = labels.squeeze(1).long()

            loss = self._criterion(outputs, labels)
            loss.backward()
            self._optimiser.step()

            _loss += loss.item() * features.size(0)
            _total += labels.numel()

        return _loss / _total

    def _epoch_valid(self, dataloader: DataLoader | TorchDataLoader) -> tuple[float, float]:
        """ Validate the model for one epoch
        :param dataloader: DataLoader for validation data
        :return: average validation loss for the epoch
        """
        # Set model to evaluation mode
        self._model.eval()

        _loss: float = 0.0
        _correct: float = 0.0
        _total: float = 0.0
        with no_grad():
            for features, labels in dataloader:
                features, labels = features.to(device(self._accelerator)), labels.to(device(self._accelerator))

                outputs = self._model(features)
                # print(outputs.shape, masks.shape)

                # Transform original size (B, 1, H, W) into new size (B, H, W) because of using CrossEntropyLoss
                labels = labels.squeeze(1).long()

                loss = self._criterion(outputs, labels)

                _loss += loss.item() * features.size(0)
                _correct += self._get_accuracy(outputs, labels)
                _total += labels.numel()

        return _loss / _total, _correct / _total

    @staticmethod
    def _get_accuracy(outputs: Tensor, labels: Tensor) -> float:
        """ Get accuracy of the model """
        predictions = outputs.argmax(dim=1)
        return predictions.eq(labels).sum().item()

    def fit(self,
            train_loader: DataLoader | TorchDataLoader, valid_loader: DataLoader | TorchDataLoader,
            epochs: int, model_save_path: str | None = None
            ) -> None:
        """ Fit the model to the training data
        :param train_loader: DataLoader for training data
        :param valid_loader: DataLoader for validation data
        :param epochs: number of training epochs
        :param model_save_path: path to save the best model parameters
        :return: None
        """
        _best_valid_loss = float("inf")
        _patience = 4
        _patience_counter = 0
        _min_delta = 5e-4

        for epoch in range(epochs):
            train_loss = self._epoch_train(train_loader)
            valid_loss, accuracy = self._epoch_valid(valid_loader)

            # Emit training and validation progress signal
            self.losses.emit(epoch + 1, train_loss, valid_loss, accuracy)

            print(f"Epoch [{epoch + 1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Valid Loss: {valid_loss:.4f} - "
                  f"Accuracy: {accuracy:.2%}")

            # Save the model if it has the best validation loss so far
            if valid_loss < _best_valid_loss - _min_delta:
                _patience_counter = 0
                _best_valid_loss = valid_loss
                save(self._model.state_dict(), model_save_path)
                print(f"Model's parameters saved to {model_save_path}")
            else:
                _patience_counter += 1
                print(f"Validation loss [{_patience_counter}/{_patience}]did not improve.")
                if _patience_counter >= _patience:
                    print(f"Early stopping triggered at the {epoch} epoch and the loss is {_best_valid_loss:4f}.")
                    break

        if _patience_counter < _patience:
            print(f"Training completed after {epochs} epochs.")


if __name__ == "__main__":
    pass
