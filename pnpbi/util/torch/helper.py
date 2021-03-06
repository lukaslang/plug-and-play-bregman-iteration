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
"""Provides helper functions to set up problems in pytorch."""
import numpy as np
import torch
from pnpbi.util.torch import functionals
from pnpbi.util.torch import operators
from pnpbi.util import radon


def setup_denoising_problem(image_size: tuple):
    """Set up denoising problem."""

    # Define identity operator for denoising.
    def K(x: torch.Tensor):
        """Identity operator."""
        return x

    # Create data fidelity and its gradient.
    G, gradG = functionals.OpSqNormDataTerm(K, K)

    return K, K, G, gradG, image_size

def setup_reconstruction_problem(image_size: tuple):
    """Set up reconstruction problem using Radon transform."""
    # Define angles.
    nangles = 180
    angles = np.linspace(0, np.pi, nangles, False)

    # Define Radon transform and adjoint.
    R, Radj, ndet = radon.radon2d(*image_size, angles)
    data_size = (nangles, ndet)

    # Create instances for use with torch.
    K = radon.RadonTransform(R, Radj, data_size)
    Kadj = radon.BackProjection(R, Radj, image_size)

    # Create data fidelity and its gradient.
    G, gradG = functionals.OpSqNormDataTerm(K, Kadj)

    return K, Kadj, G, gradG, data_size
