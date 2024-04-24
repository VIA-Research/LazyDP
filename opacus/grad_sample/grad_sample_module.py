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

from __future__ import annotations

import logging
import warnings
from functools import partial
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from opacus.grad_sample.functorch import ft_compute_per_sample_gradient, prepare_layer
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.layers.dp_rnn import DPGRU, DPLSTM, DPRNN, RNNLinear
from opacus.utils.module_utils import (
    has_trainable_params,
    requires_grad,
    trainable_modules,
    trainable_parameters,
)
import config
from config import MODE_DPSGD_B, MODE_DPSGD_R, MODE_DPSGD_F, MODE_LAZYDP, MODE_EANA

logger = logging.getLogger(__name__)


def create_or_accumulate_grad_sample(
    *, param: torch.Tensor, grad_sample: torch.Tensor, max_batch_len: int
) -> None:
    """
    Creates a ``_current_grad_sample`` attribute in the given parameter, or adds to it
    if the ``_current_grad_sample`` attribute already exists.


    Args:
        param: Parameter to which ``grad_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
        layer: nn.Module parameter belongs to
    """
    if param.requires_grad:
        if hasattr(param, "_current_grad_sample"):
            assert False, "Disable RNNs"
            param._current_grad_sample[: grad_sample.shape[0]] += grad_sample
        else:
            if config.dpsgd_mode == MODE_DPSGD_B:
                assert(param.grad_sample is None)
            param.grad_sample = grad_sample
            # param._current_grad_sample = torch.zeros(
            #     torch.Size([max_batch_len]) + grad_sample.shape[1:],
            #     device=grad_sample.device,
            #     dtype=grad_sample.dtype,
            # )
            # param._current_grad_sample[: grad_sample.shape[0]] = grad_sample


def promote_current_grad_sample(p: nn.Parameter) -> None:
    if p.requires_grad:
        if p.grad_sample is not None:
            return
            if isinstance(p.grad_sample, list):
                p.grad_sample.append(p._current_grad_sample)
            else:
                p.grad_sample = [p.grad_sample, p._current_grad_sample]
        else:
            assert False
            p.grad_sample = p._current_grad_sample

        del p._current_grad_sample


class GradSampleModule(AbstractGradSampleModule):
    """
    Hooks-based implementation of AbstractGradSampleModule

    Computes per-sample gradients using custom-written methods for each layer.
    See README.md for more details
    """

    GRAD_SAMPLERS = {}

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
    ):
        """

        Args:
            m: nn.Module to be wrapped
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            strict: If set to ``True``, the input module will be validated to check that
                ``GradSampleModule`` has grad sampler functions for all submodules of
                the input module (i.e. if it knows how to calculate per sample gradients)
                for all model parameters. If set to ``False``, per sample gradients will
                be computed on "best effort" basis - they will be available where
                possible and set to None otherwise. This is not recommended, because
                some unsupported modules (e.g. BatchNorm) affect other parameters and
                invalidate the concept of per sample gradients for the entire model.
            force_functorch: If set to ``True``, will use functorch to compute
                all per sample gradients. Otherwise, functorch will be used only
                for layers without registered grad sampler methods.

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) doesn't have a registered grad sampler function.
        """
        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )

        errors = self.validate(module=m, strict=strict)
        if errors and not strict:
            logger.info(
                f"GradSampleModule found the following errors: {errors}."
                "Using non-strict mode, continuing"
            )

        self.hooks_enabled = False
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self.force_functorch = force_functorch
        self.add_hooks(
            loss_reduction=loss_reduction,
            batch_first=batch_first,
            force_functorch=force_functorch,
        )

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def iterate_submodules(self, module: nn.Module) -> Iterable[nn.Module]:
        if has_trainable_params(module):
            yield module

        # Don't recurse if module is handled by functorch
        if (
            has_trainable_params(module)
            and type(module) not in self.GRAD_SAMPLERS
            and type(module) not in [DPRNN, DPLSTM, DPGRU]
        ):
            return

        for m in module.children():
            yield from self.iterate_submodules(m)

    def add_hooks(
        self,
        *,
        loss_reduction: str = "mean",
        batch_first: bool = True,
        force_functorch: bool = False,
    ) -> None:
        """
        Adds hooks to model to save activations and backprop values.
        The hooks will
        1. save activations into param.activations during forward pass
        2. compute per-sample gradients in params.grad_sample during backward pass.
        Call ``remove_hooks(model)`` to disable this.

        Args:
            model: the model to which hooks are added
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            force_functorch: If set to ``True``, will use functorch to compute all per sample gradients.
                Otherwise, functorch will be used only for layers without registered grad sampler methods.
        """
        if hasattr(self._module, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")
        else:
            self._module.autograd_grad_sample_hooks = []
            self.autograd_grad_sample_hooks = self._module.autograd_grad_sample_hooks

        for module in self.iterate_submodules(self._module):
            # Do not add hooks to DPRNN, DPLSTM or DPGRU as the hooks are handled by the `RNNLinear`
            if type(module) in [DPRNN, DPLSTM, DPGRU]:
                continue

            if force_functorch or not type(module) in self.GRAD_SAMPLERS:
                prepare_layer(module, batch_first=batch_first)

            self.autograd_grad_sample_hooks.append(
                module.register_forward_hook(self.capture_activations_hook)
            )

            self.autograd_grad_sample_hooks.append(
                module.register_backward_hook(
                    partial(
                        self.capture_backprops_hook,
                        loss_reduction=loss_reduction,
                        batch_first=batch_first,
                    )
                )
            )

        self.enable_hooks()

    def remove_hooks(self) -> None:
        """
        Removes hooks added by ``add_hooks()``
        """
        self.disable_hooks()

        for p in self.parameters():
            if hasattr(p, "ddp_hooks"):
                while p.ddp_hooks:
                    handle = p.ddp_hooks.pop()
                    handle.remove()
                delattr(p, "ddp_hooks")

        if not hasattr(self, "autograd_grad_sample_hooks"):
            raise ValueError("Asked to remove hooks, but no hooks found")
        else:
            while self.autograd_grad_sample_hooks:
                handle = self.autograd_grad_sample_hooks.pop()
                handle.remove()
            delattr(self, "autograd_grad_sample_hooks")
            delattr(self._module, "autograd_grad_sample_hooks")

        # Remove functorch hooks
        for _module_name, module in trainable_modules(self._module):
            if hasattr(module, "ft_compute_sample_grad"):
                delattr(module, "ft_compute_sample_grad")

    def disable_hooks(self) -> None:
        r"""
        Globally disable all hooks installed by this library.
        Why is this needed? As per https://github.com/pytorch/pytorch/issues/25723, there is
        a bug in Autograd that makes removing hooks do nothing if the graph was already
        constructed. For this reason, we have this method to at least turn them off.
        """
        self.hooks_enabled = False

    def enable_hooks(self) -> None:
        r"""
        The opposite of ``disable_hooks()``. Hooks are always enabled unless you explicitly
        disable them so you don't need to call this unless you want to re-enable them.
        """
        self.hooks_enabled = True

    def _close(self):
        super()._close()
        self.remove_hooks()

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        if (
            not requires_grad(module)
            or not module.training
            or not torch.is_grad_enabled()
        ):
            return

        if not self.hooks_enabled:
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append([t.detach() for t in forward_input])  # pyre-ignore
        
        for _, p in trainable_parameters(module):
            p._forward_counter += 1
        

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        """
        Computes per sample gradients given the current backprops and activations
        stored by the associated forward hook. Computed per sample gradients are
        stored in ``grad_sample`` field in each parameter.

        For non-recurrent layers the process is straightforward: for each
        ``loss.backward()`` call this hook will be called exactly one. For recurrent
        layers, however, this is more complicated and the hook will be called multiple
        times, while still processing the same batch of data.

        For this reason we first accumulate the gradients from *the same batch* in
        ``p._current_grad_sample`` and then, when we detect the end of a full backward
        pass - we store accumulated result on ``p.grad_sample``.

        From there, ``p.grad_sample`` could be either a Tensor or a list of Tensors,
        if accumulated over multiple batches

        Args:
            module: nn.Module,
            _forward_input: torch.Tensor,
            forward_output: torch.Tensor,
            loss_reduction: str,
            batch_first: bool,
        """

        if not self.hooks_enabled:
            return

        backprops = forward_output[0].detach()
        activations, backprops = self.rearrange_grad_samples(
            module=module,
            backprops=backprops,
            loss_reduction=loss_reduction,
            batch_first=batch_first,
        )
        if not self.force_functorch and type(module) in self.GRAD_SAMPLERS:
            grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
        else:
            assert False, "Exclude self.force_functorch"
            grad_sampler_fn = ft_compute_per_sample_gradient
        
        if config.dpsgd_mode in [MODE_DPSGD_B, MODE_DPSGD_R]:
            grad_samples = grad_sampler_fn(module, activations, backprops)
            for param, gs in grad_samples.items():
                create_or_accumulate_grad_sample(
                    param=param, grad_sample=gs, max_batch_len=module.max_batch_len
                )

            # Detect end of current batch processing and switch accumulation
            # mode from sum to stacking. Used for RNNs and tied parameters
            # (See #417 for details)

            for _, p in trainable_parameters(module):
                p._forward_counter -= 1
                if p._forward_counter == 0:
                    promote_current_grad_sample(p)
        
        # For reweight DP-SGD, compute norm of each parameters
        if config.dpsgd_mode == MODE_DPSGD_R:
            for _, p in trainable_parameters(module):
                assert p.requires_grad == True
                p.grad_sample_norms = [p.grad_sample.norm(2, dim=list(range(1, len(p.grad_sample.shape))))]
                del p.grad_sample
        elif config.dpsgd_mode in [MODE_DPSGD_F, MODE_LAZYDP, MODE_EANA]:
            # input norm x output gradients norm = per-sample gradient norms
            if type(module) == nn.Linear:
                activations_norm = activations[0].norm(2, dim=-1)
            backprops_norm = backprops.norm(2, dim=-1)
            for _, p in trainable_parameters(module):
                assert p.requires_grad == True
                if type(module) == nn.Linear:
                    if p is module.weight:
                        p.grad_sample_norms = [activations_norm * backprops_norm]
                    elif p is module.bias:
                        p.grad_sample_norms = [backprops_norm]
                    else:
                        assert False, "Never happen"
                elif type(module) == nn.EmbeddingBag:
                    assert config.cur_batch_size == len(activations[2])
                    cur_num_gathers = activations[2][1].item() - activations[2][0].item()
                    assert min(config.num_gathers, p.shape[0]) == cur_num_gathers
                    
                    p.grad_sample_norms = [backprops_norm * (cur_num_gathers**0.5)]
                else:
                    assert False, "unknown layer"
            
                
            
            #p.grad_sample_norms = list(activations_norm * backprops_norm + backprops_norm)

        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len


    def rearrange_grad_samples(
        self,
        *,
        module: nn.Module,
        backprops: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rearrange activations and grad_samples based on loss reduction and batch dim

        Args:
            module: the module for which per-sample gradients are computed
            backprops: the captured backprops
            loss_reduction: either "mean" or "sum" depending on whether backpropped
                loss was averaged or summed over batch
            batch_first: True is batch dimension is first
        """
        if not hasattr(module, "activations"):
            raise ValueError(
                f"No activations detected for {type(module)},"
                " run forward after add_hooks(model)"
            )

        batch_dim = 0 if batch_first or type(module) is RNNLinear else 1

        if not hasattr(module, "max_batch_len"):
            # For packed sequences, max_batch_len is set in the forward of the model (e.g. the LSTM)
            # Otherwise we infer it here
            module.max_batch_len = _get_batch_size(
                module=module,
                batch_dim=batch_dim,
            )
        activations = module.activations.pop()

        n = module.max_batch_len
        if loss_reduction == "mean":
            backprops = backprops * n
        elif loss_reduction == "sum":
            backprops = backprops
        else:
            raise ValueError(
                f"loss_reduction = {loss_reduction}. Only 'sum' and 'mean' losses are supported"
            )

        # No matter where the batch dimension was, .grad_samples will *always* put it in the first dim
        if batch_dim != 0:
            activations = [
                t.permute([batch_dim] + [x for x in range(t.dim()) if x != batch_dim])
                for t in activations
            ]
            backprops = backprops.permute(
                [batch_dim] + [x for x in range(backprops.dim()) if x != batch_dim]
            )

        return activations, backprops

    @classmethod
    def is_supported(cls, module: nn.Module) -> bool:
        """
        Checks if this individual model is supported (i.e. has a registered
        grad sampler function)

        Notes:
            Note that this method does not check submodules

        Args:
            module: nn.Module to be checked

        Returns:
            ``True`` if grad sampler is found, ``False`` otherwise
        """
        warnings.warn(
            "GradSampleModule.is_supported is deprecated, as all layers can now be used with functorch.",
            DeprecationWarning,
        )

        return True

    @classmethod
    def validate(
        cls, module: nn.Module, *, strict: bool = False
    ) -> List[NotImplementedError]:
        """
        Check if per sample gradients can be fully computed for a given model

        Args:
            module: nn.Module to be checked
            raise_if_error: Behaviour in case of a negative check result. Will
            return the list of exceptions if set to ``False``, and throw otherwise

        Returns:
            Empty list of validation is successful.
            List of validation errors  if ``raise_if_error=False`` and
            unsupported modules are found

        Raises:
            NotImplementedError
                If ``raise_if_error=True`` and unsupported modules are found
        """
        errors = []
        errors.extend(
            [
                NotImplementedError(
                    f"Model contains a trainable layer "
                    f"that Opacus doesn't currently support({m_name}:{m}). "
                    f"Please implement and register grad sampler for this layer. "
                    f"(See opacus.grad_sample.utils.register_grad_sampler)"
                )
                for m_name, m in trainable_modules(module)
                # With functorch, all modules are trainable
                # We still want to avoid module that have buffers (e.g. BatchNorm)
                # as the buffers are not private
                if len(list(m.buffers())) > 0
            ]
        )
        # raise or return errors as needed
        if strict and len(errors) > 0:
            raise NotImplementedError(errors)
        else:
            return errors


def _get_batch_size(*, module: nn.Module, batch_dim: int) -> int:
    """
    Computes and returns the maximum batch size which is the maximum of the dimension values
    along 'batch_dim' axis over module.activations, where module.activations is
    a list.

    Args:
        module: input module
        batch_dim: batch dimension

    Returns:
        Maximum sequence length in a batch
    """
    max_batch_len = 0
    for out in module.activations:
        # out is typically a tuple of one element (x)
        # for embedding bag, it is a tuple of two elements (x, offsets)
        # where len(offsets) = batch_size
        if out[-1].shape[batch_dim] > max_batch_len:
            max_batch_len = out[-1].shape[batch_dim]

    return max_batch_len
