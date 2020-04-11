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
"""Encapsulates use of ASTRA Toolbox."""
import astra
import numpy as np


def radon2d(m: int, n: int, angles: np.array, cuda=False):
    """Create operator for the 2D Radon transform.

    K takes an array of shape (m, n) and returns an array of shape
    (nangles, ndet), where nangles is the number of measurement angles, i.e.
    angles.size, and ndet is the number of detectors, calculated as:

    >>> ndet = np.ceil(np.hypot(m, n)).astype(int)

    Args:
    ----
        m (int): Number of rows. m > 1.
        n (int): Number of columns. n > 1.
        angles (np.array): An array specifying measurement angles in rad in the
        range [0, pi].
        cuda (bool): Uses GPU (default is device 0) if True.

    Return:
    ------
        K: A function that takes an array of shape (m, n).
        Kadj: A function that takes an array of shape (nangles, ndet).
        ndet: The number of detectors used.
    """
    ndet = np.ceil(1.5 * np.hypot(m, n)).astype(int)
    det_spacing = 1
    vol_geom = astra.create_vol_geom(m, n)
    proj_geom = astra.create_proj_geom('parallel',
                                       det_spacing, ndet, angles)

    proj_type = 'cuda' if cuda else 'linear'
    alg = 'BP_CUDA' if cuda else 'BP'

    def K(x):
        # Create sinogram.
        proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)
        sino_id, sino = astra.create_sino(x, proj_id)
        astra.data2d.delete(sino_id)
        astra.projector.delete(proj_id)
        return sino

    def Kadj(sino):
        # Create objects for the reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom)
        sino_id = astra.data2d.create('-sino', proj_geom, sino)
        proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)

        # Set up the parameters for the backpropagation reconstruction.
        cfg = astra.astra_dict(alg)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ProjectorId'] = proj_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)

        # Run algorithm.
        astra.algorithm.run(alg_id)

        # Get the result
        rec = astra.data2d.get(rec_id)

        # Clean up and return result.
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)
        astra.projector.delete(proj_id)
        return rec

    return K, Kadj, ndet
