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
"""Provides helpers to encapsulate operators in PyTorch."""
import torch


class LinearOperator(torch.autograd.Function):
    """An linear operator."""

    @staticmethod
    def forward(ctx, x, K, Kadj):
        """Compute the forward pass for given put data.

        Args:
        ----
            ctx: The context.
            x (Tensor): The input.
            K, Kadj: Function handles for a linear operator and its adjoint.

        Return:
        ------
            result (Tensor): The result of applying K to x.
        """
        ctx.Kadj = Kadj
        result = x.new(K(x.detach().numpy()))
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the backward pass for given gradient output.

        Args:
        ----
            ctx: The context.
            grad_output (Tensor): The gradient output.

        Return:
        ------
            result: A 3-tuple with the first element being the gradient with
            respect to the input.
        """
        x, = ctx.saved_tensors
        output = grad_output.numpy()
        return x.new(ctx.Kadj(output)), None, None
