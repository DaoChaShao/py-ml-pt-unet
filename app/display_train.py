#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/31 16:38
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   display_train.py
# @Desc     :   

from random import randint
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel)
from sys import argv, exit

from utils.config import CONFIG
from utils.helper import load_data_paths


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UNet Train Data Display")
        self.resize(800, 400)
        self._widget = QWidget()
        self.setCentralWidget(self._widget)

        self._lbl_names: list[str] = ["Images", "Masks"]
        self._labels: list[QLabel] = []
        self._btn_names: list[str] = ["Display", "Clear", "Quit"]
        self._buttons: list[QPushButton] = []

        self._setup()

    def _setup(self):
        _layout = QVBoxLayout(self._widget)
        _row_lbl = QHBoxLayout()
        _row_btn = QHBoxLayout()

        for name in self._lbl_names:
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            label.setMaximumSize(400, 400)
            self._labels.append(label)
            _row_lbl.addWidget(label)
        _layout.addLayout(_row_lbl)

        funcs = [
            self._display,
            self._clear,
            self._quit,
        ]
        for i, func in enumerate(funcs):
            button = QPushButton(self._btn_names[i])
            button.clicked.connect(func)
            match button.text():
                case "Clear":
                    button.setEnabled(False)
            self._buttons.append(button)
            _row_btn.addWidget(button)
        _layout.addLayout(_row_btn)

    def _display(self):
        # Get data paths
        paths = load_data_paths(Path(CONFIG.FILEPATHS.DATASET_TRAIN))
        print(len(paths[0]), paths[0][:3])
        print(len(paths[1]), paths[1][:3])
        # Get image and mask path at random index
        index: int = randint(0, len(paths[0]) - 1)
        print(paths[0][index])
        print(paths[1][index])

        for lbl in self._labels:
            match lbl.text():
                case "Images":
                    pixmap = QPixmap(paths[0][index])
                    lbl.setPixmap(pixmap)
                    lbl.setScaledContents(True)
                case "Masks":
                    lbl.setPixmap(QPixmap(paths[1][index]))
                    lbl.setScaledContents(True)

        for btn in self._buttons:
            match btn.text():
                case "Display":
                    btn.setEnabled(False)
                case "Clear":
                    btn.setEnabled(True)

    def _clear(self):
        for name, lbl in zip(self._lbl_names, self._labels):
            lbl.setText(name)

        for btn in self._buttons:
            match btn.text():
                case "Display":
                    btn.setEnabled(True)
                case "Clear":
                    btn.setEnabled(False)

    def _quit(self):
        self.close()


if __name__ == "__main__":
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())
