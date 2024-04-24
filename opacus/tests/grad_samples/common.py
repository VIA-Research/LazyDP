#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import unittest
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import wrap_model
from opacus.utils.module_utils import trainable_parameters
from opacus.utils.packed_sequences import compute_seq_lengths
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.testing import assert_allclose


def expander(x, factor: int = 2):
    return x * factor


def shrinker(x, factor: int = 2):
    return max(1, x // factor)  # if avoid returning 0 for x == 1


def is_batch_empty(batch: Union[torch.Tensor, Iterable[torch.Tensor]]):
    if type(batch) is torch.Tensor:
        return batch.numel() == 0
    else:
        return batch[0].numel() == 0


class ModelWithLoss(nn.Module):
    """
    To test the gradients of a module, we need to have a loss.
    This module makes it easy to get a loss from any nn.Module, and automatically generates
    a target y vector for it in the forward (of all zeros of the correct size).
    This reduces boilerplate while testing.
    """

    supported_reductions = ["mean", "sum"]

    def __init__(self, module: nn.Module, loss_reduction: str = "mean"):
        """
        Instantiates this module.

        Args:
            module: The nn.Module you want to test.
            loss_reduction: What reduction to apply to the loss. Defaults to "mean".

        Raises:
            ValueError: If ``loss_reduction`` is not among those supported.
        """
        super().__init__()
        self.wrapped_module = module

        if loss_reduction not in self.supported_reductions:
            raise ValueError(
                f"Passed loss_reduction={loss_reduction}. Only {self.supported_reductions} supported."
            )
        self.criterion = nn.L1Loss(reduction=loss_reduction)

    def forward(self, x):
        if type(x) is tuple:
            x = self.wrapped_module(*x)
        else:
            x = self.wrapped_module(x)
        if type(x) is PackedSequence:
            loss = _compute_loss_packedsequences(self.criterion, x)
        else:
            y = torch.zeros_like(x)
            loss = self.criterion(x, y)
        return loss


def clone_module(module: nn.Module) -> nn.Module:
    """
    Handy utility to clone an nn.Module. PyTorch doesn't always support copy.deepcopy(), so it is
    just easier to serialize the model to a BytesIO and read it from there.

    Args:
        module: The module to clone

    Returns:
        The clone of ``module``
    """
    with io.BytesIO() as bytesio:
        torch.save(module, bytesio)
        bytesio.seek(0)
        module_copy = torch.load(bytesio)
    return module_copy


class GradSampleHooks_test(unittest.TestCase):
    """
    Set of common testing utils. It is meant to be subclassed by your test.
    See other tests as an example of how this is done.
    """

    def compute_microbatch_grad_sample(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        module: nn.Module,
        batch_first=True,
        loss_reduction="mean",
        chunk_method=iter,
    ) -> Dict[str, torch.tensor]:
        """
        Computes per-sample gradients with the microbatch method, i.e. by computing normal gradients
        with batch_size set to 1, and manually accumulating them. This is our reference for testing
        as this method is obviously correct, but slow.

        Args:
            x: The tensor in input to the ``module``
            module: The ``ModelWithLoss`` that wraps the nn.Module you want to test.
            batch_first: Whether batch size is the first dimension (as opposed to the second).
                Defaults to True.
            loss_reduction: What reduction to apply to the loss. Defaults to "mean".
            chunk_method: The method to use to split the batch into microbatches. Defaults to ``iter``.

        Returns:
            Dictionary mapping parameter_name -> per-sample-gradient for that parameter
        """
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        np.random.seed(0)

        module = ModelWithLoss(clone_module(module), loss_reduction)

        for _, p in trainable_parameters(module):
            p.microbatch_grad_sample = []

        if not batch_first and type(x) is not list:
            # This allows us to iterate with x_i
            x = x.transpose(0, 1)

        # Invariant: x is [B, T, ...]

        for x_i in chunk_method(x):
            # x_i is [T, ...]
            module.zero_grad()
            if type(x_i) is not tuple:
                # EmbeddingBag provides tuples
                x_i = x_i.unsqueeze(
                    0 if batch_first else 1
                )  # x_i of size [1, T, ...] if batch_first, else [T, 1, ...]
            loss_i = module(x_i)
            loss_i.backward()
            for p in module.parameters():
                p.microbatch_grad_sample.append(p.grad.detach().clone())

        for _, p in trainable_parameters(module):
            if batch_first:
                p.microbatch_grad_sample = torch.stack(
                    p.microbatch_grad_sample, dim=0  # [B, T, ...]
                )
            else:
                p.microbatch_grad_sample = torch.stack(
                    p.microbatch_grad_sample, dim=1  # [T, B, ...]
                ).transpose(
                    0, 1
                )  # Opacus's semantics is that grad_samples are ALWAYS batch_first: [B, T, ...]

        microbatch_grad_samples = {
            name: p.microbatch_grad_sample
            for name, p in trainable_parameters(module.wrapped_module)
        }
        return microbatch_grad_samples

    def compute_opacus_grad_sample(
        self,
        x: Union[torch.Tensor, PackedSequence],
        module: nn.Module,
        batch_first=True,
        loss_reduction="mean",
        grad_sample_mode="hooks",
    ) -> Dict[str, torch.tensor]:
        """
        Runs Opacus to compute per-sample gradients and return them for testing purposes.

        Args:
            x: The tensor in input to the ``module``
            module: The ``ModelWithLoss`` that wraps the nn.Module you want to test.
            batch_first: Whether batch size is the first dimension (as opposed to the second).
                Defaults to True.
            loss_reduction: What reduction to apply to the loss. Defaults to "mean".

        Returns:
            Dictionary mapping parameter_name -> per-sample-gradient for that parameter
        """
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        np.random.seed(0)

        gs_module = wrap_model(
            model=clone_module(module),
            grad_sample_mode=grad_sample_mode,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )
        grad_sample_module = ModelWithLoss(gs_module, loss_reduction)

        grad_sample_module.zero_grad()
        loss = grad_sample_module(x)
        loss.backward()

        opacus_grad_samples = {
            name: p.grad_sample
            for name, p in trainable_parameters(
                grad_sample_module.wrapped_module._module
            )
        }

        return opacus_grad_samples

    def run_test(
        self,
        x: Union[torch.Tensor, PackedSequence, Tuple],
        module: nn.Module,
        batch_first=True,
        atol=10e-6,
        rtol=10e-5,
        ew_compatible=True,
        chunk_method=iter,
    ):
        grad_sample_modes = ["hooks", "functorch"]
        try:
            import functorch  # noqa
        except ImportError:
            grad_sample_modes = ["hooks"]

        if type(module) is nn.EmbeddingBag or (
            type(x) is not PackedSequence and is_batch_empty(x)
        ):
            grad_sample_modes = ["hooks"]

        for grad_sample_mode in grad_sample_modes:
            for loss_reduction in ["sum", "mean"]:

                with self.subTest(
                    grad_sample_mode=grad_sample_mode, loss_reduction=loss_reduction
                ):
                    self.run_test_with_reduction(
                        x,
                        module,
                        batch_first=batch_first,
                        loss_reduction=loss_reduction,
                        atol=atol,
                        rtol=rtol,
                        grad_sample_mode=grad_sample_mode,
                        chunk_method=chunk_method,
                    )
        if ew_compatible and batch_first and torch.__version__ >= (1, 13):
            self.run_test_with_reduction(
                x,
                module,
                batch_first=batch_first,
                loss_reduction="sum",
                atol=atol,
                rtol=rtol,
                grad_sample_mode="ew",
                chunk_method=chunk_method,
            )

    def run_test_with_reduction(
        self,
        x: Union[torch.Tensor, PackedSequence],
        module: nn.Module,
        batch_first=True,
        loss_reduction="mean",
        atol=10e-6,
        rtol=10e-5,
        grad_sample_mode="hooks",
        chunk_method=iter,
    ):
        opacus_grad_samples = self.compute_opacus_grad_sample(
            x,
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
        )

        if type(x) is PackedSequence:
            x_unpacked = _unpack_packedsequences(x)
            microbatch_grad_samples = self.compute_microbatch_grad_sample(
                x_unpacked,
                module,
                batch_first=batch_first,
                loss_reduction=loss_reduction,
            )
        elif not is_batch_empty(x):
            microbatch_grad_samples = self.compute_microbatch_grad_sample(
                x,
                module,
                batch_first=batch_first,
                loss_reduction=loss_reduction,
                chunk_method=chunk_method,
            )
        else:
            # We've checked opacus can handle 0-sized batch. Microbatch doesn't make sense
            return

        if microbatch_grad_samples.keys() != opacus_grad_samples.keys():
            raise ValueError(
                "Keys not matching! "
                f"Keys only in microbatch: {microbatch_grad_samples.keys() - opacus_grad_samples.keys()}; "
                f"Keys only in Opacus: {opacus_grad_samples.keys() - microbatch_grad_samples.keys()}"
            )

        self.check_shapes(microbatch_grad_samples, opacus_grad_samples, loss_reduction)
        self.check_values(
            microbatch_grad_samples, opacus_grad_samples, loss_reduction, atol, rtol
        )

    def check_shapes(
        self,
        microbatch_grad_samples,
        opacus_grad_samples,
        loss_reduction,
    ) -> None:
        failed = []
        for name, opacus_grad_sample in opacus_grad_samples.items():
            microbatch_grad_sample = microbatch_grad_samples[name]
            msg = (
                f"Param '{name}': "
                f"from Opacus: {opacus_grad_sample.shape}, "
                f"from Microbatch: {microbatch_grad_sample.shape}. "
            )
            try:
                self.assertEqual(
                    opacus_grad_sample.shape,
                    microbatch_grad_sample.shape,
                    msg=msg,
                )

            except AssertionError:
                failed.append(msg)

        if failed:
            failed_str = "\n\t".join(f"{i}. {s}" for i, s in enumerate(failed, 1))
            raise AssertionError(
                f"A total of {len(failed)} shapes do not match "
                f"for loss_reduction={loss_reduction}: \n\t{failed_str}"
            )

    def check_values(
        self,
        microbatch_grad_samples,
        opacus_grad_samples,
        loss_reduction,
        atol,
        rtol,
    ) -> None:
        failed = []
        for name, opacus_grad_sample in opacus_grad_samples.items():
            microbatch_grad_sample = microbatch_grad_samples[name]
            msg = (
                f"Param {name}: Opacus L2 norm = : {opacus_grad_sample.norm(2)}, ",
                f"Microbatch L2 norm = : {microbatch_grad_sample.norm(2)}, ",
                f"MSE = {F.mse_loss(opacus_grad_sample, microbatch_grad_sample)}, ",
                f"L1 Loss = {F.l1_loss(opacus_grad_sample, microbatch_grad_sample)}",
            )
            try:
                assert_allclose(
                    actual=microbatch_grad_sample,
                    expected=opacus_grad_sample,
                    atol=atol,
                    rtol=rtol,
                )
            except AssertionError:
                failed.append(msg)
        if failed:
            failed_str = "\n\t".join(f"{i}. {s}" for i, s in enumerate(failed, 1))
            raise AssertionError(
                f"A total of {len(failed)} values do not match "
                f"for loss_reduction={loss_reduction}: \n\t{failed_str}"
            )


def _unpack_packedsequences(X: PackedSequence) -> List[torch.Tensor]:
    r"""
    Produces a list of tensors from X (PackedSequence) such that this list was used to create X with batch_first=True

    Args:
        X: A PackedSequence from which the output list of tensors will be produced.

    Returns:
        unpacked_data: The list of tensors produced from X.
    """

    X_padded = pad_packed_sequence(X)
    X_padded = X_padded[0].permute((1, 0, 2))

    if X.sorted_indices is not None:
        X_padded = X_padded[X.sorted_indices]

    seq_lens = compute_seq_lengths(X.batch_sizes)
    unpacked_data = [0] * len(seq_lens)
    for idx, length in enumerate(seq_lens):
        unpacked_data[idx] = X_padded[idx][:length, :]

    return unpacked_data


def _compute_loss_packedsequences(
    criterion: nn.L1Loss, x: PackedSequence
) -> torch.Tensor:
    r"""
    This function computes the loss in a different way for 'mean' reduced L1 loss while for 'sum' reduced L1 loss,
    it computes the same way as with non-packed data. For 'mean' reduced L1 loss, it transforms x (PackedSequence)
    into a list of tensors such that this list of tensors was used to create this PackedSequence in the first
    place using batch_first=True and then takes the mean of the loss values produced from applying criterion on
    each sequence sample.

    Args:
        criterion: An L1 loss function with reduction either set to 'sum' or 'mean'.
        x: Data in the form of a PackedSequence.

    Returns:
        A loss variable, reduced either using summation or averaging from L1 errors.
    """

    if criterion.reduction == "sum":
        y = torch.zeros_like(x[0])
        return criterion(x[0], y)
    elif criterion.reduction == "mean":
        x = _unpack_packedsequences(x)
        loss_sum = 0
        for x_i in x:
            y_i = torch.zeros_like(x_i)
            loss_sum += criterion(x_i, y_i)
        loss_mean = loss_sum / len(x)
        return loss_mean
