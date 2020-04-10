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
import numpy as np
from pnpbi.util.utils import RunningAvg
import unittest


class TestUtils(unittest.TestCase):

    def test_running_average(self):

        loss_avg = RunningAvg()

        # Test if no value has been added.
        np.testing.assert_allclose(loss_avg(), 0)

        # Test if values has been added.
        loss_avg.add(2)
        loss_avg.add(3)
        np.testing.assert_allclose(loss_avg(), 2.5)


if __name__ == '__main__':
    unittest.main()
