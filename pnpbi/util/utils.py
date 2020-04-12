#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019 Lukas Lang
#
# This file is part of PNPBI.
#
#    PNPBI is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PNPBI is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PNPBI. If not, see <http://www.gnu.org/licenses/>.
"""Provides utils."""
import json
import logging
import os
import torch


class RunningAvg():
    """A class that maintains the running average.

    Example:
    -------
    ```
    loss_avg = RunningAvg()
    loss_avg.add(2)
    loss_avg.add(4)
    print(loss_avg())
    ```

    The example will output 3.

    loss_avg() will return 0 in case add hasn't been called.
    """

    def __init__(self):
        self.count = 0
        self.sum = 0

    def add(self, value):
        """Add value to the running average."""
        self.count += 1
        self.sum += value

    def __call__(self):
        """Return the average."""
        return self.sum / self.count if self.count > 0 else 0


class Params():
    """Helper class that loads/saves hyperparameters to/from a JSON file."""

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """Save to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Load params."""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __str__(self):
        """Return string representation of parameters."""
        return json.dumps(self.__dict__, indent=2)

    @property
    def dict(self):
        """Return dict to access parameters as params.dict['xyz']."""
        return self.__dict__


def set_logger(log_path):
    """Initialize logger to write to file and to terminal.

    Example:
    -------
    ```
    log = utils.set_logger(log_path)
    log.info("Log message...")
    ```

    Args:
    ----
        log_path (str): A valid path.

    Return:
    ------
        logger: The logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def load_checkpoint(restore_file, model, optimizer=None, device=None):
    """Load model parameters (state_dict) from restore_file.

    If optimizer is specified it is assumed restore_files also holds these
    parameters and loads them parameters.

    Args:
    ----
        restore_file (string): The file from which to read state_dict.
        model (torch.nn.Module): The model for which parameters are loaded.
        optimizer (torch.optim.Optimizer): The optimizer for which parameters
            are loaded.
        device (torch.device): The target device.

    Return:
    ------
        epoch (int): The number of finished epochs.
    """
    if os.path.exists(restore_file):
        checkpoint = torch.load(restore_file) if device is None \
            else torch.load(restore_file, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])

        return checkpoint['epoch']

    logging.info(f"Checkpoint doesn't exist: '{restore_file}'.")
    return 0


def save_checkpoint(state, model_dir):
    """Save state (model and optimizer parameters).

    An existing checkpoint will be overwritten.

    Args:
    ----
        state (dict): Contains model's state_dict. May contain other entries.
        model_dir (string): Directory where checkpoint is stored.
    """
    checkpoint_file = os.path.join(model_dir, 'checkpoint.pth.tar')
    if not os.path.exists(model_dir):
        print(f"Checkpoint directory doesn't exit. Creating {model_dir}.")
        os.mkdir(model_dir)

    # Save state.
    torch.save(state, checkpoint_file)
