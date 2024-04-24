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
from typing import Callable, List, Optional, Union

import torch
from opacus.optimizers.utils import params
from opt_einsum.contract import contract
from torch import nn
from torch.optim import Optimizer

import config
from config import MODE_DPSGD_B, MODE_DPSGD_R, MODE_DPSGD_F, MODE_LAZYDP, MODE_EANA

from opacus.grad_sample import AbstractGradSampleModule, GradSampleModule
from opacus.utils.module_utils import trainable_modules, trainable_parameters

import numpy as np
import custom_api_cpp

logger = logging.getLogger(__name__)

def setdiff1d_tensor(a: torch.Tensor, b: torch.Tensor): # {a - b}
        return torch.from_numpy(np.setdiff1d(a.numpy(), b.numpy()))
    

def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Marks parameters that have already been used in the optimizer step.

    DP-SGD puts certain restrictions on how gradients can be accumulated. In particular,
    no gradient can be used twice - client must call .zero_grad() between
    optimizer steps, otherwise privacy guarantees are compromised.
    This method marks tensors that have already been used in optimizer steps to then
    check if zero_grad has been duly called.

    Notes:
        This is used to only mark ``p.grad_sample`` and ``p.summed_grad``

    Args:
        obj: tensor or a list of tensors to be marked
    """

    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True


def _check_processed_flag_tensor(x: torch.Tensor):
    """
    Checks if this gradient tensor has been previously used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor

    Raises:
        ValueError
            If tensor has attribute ``._processed`` previously set by
            ``_mark_as_processed`` method
    """

    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients haven't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )


def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Checks if this gradient tensor (or a list of tensors) has been previously
    used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor or a list of tensors

    Raises:
        ValueError
            If tensor (or at least one tensor from the list) has attribute
            ``._processed`` previously set by ``_mark_as_processed`` method
    """

    if isinstance(obj, torch.Tensor):
        _check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            _check_processed_flag_tensor(x)

def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
    secure_mode: bool = False,
) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0

    Args:
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
        secure_mode: boolean showing if "secure" noise need to be generated
            (see the notes)

    Notes:
        If `secure_mode` is enabled, the generated noise is also secure
        against the floating point representation attacks, such as the ones
        in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.
        The attack for Opacus first appeared in https://arxiv.org/abs/2112.05307.
        The implemented fix is based on https://arxiv.org/abs/2107.10138 and is
        achieved through calling the Gaussian noise function 2*n times, when n=2
        (see section 5.1 in https://arxiv.org/abs/2107.10138).

        Reason for choosing n=2: n can be any number > 1. The bigger, the more
        computation needs to be done (`2n` Gaussian samples will be generated).
        The reason we chose `n=2` is that, `n=1` could be easy to break and `n>2`
        is not really necessary. The complexity of the attack is `2^p(2n-1)`.
        In PyTorch, `p=53` and so complexity is `2^53(2n-1)`. With `n=1`, we get
        `2^53` (easy to break) but with `n=2`, we get `2^159`, which is hard
        enough for an attacker to break.
    """
    #zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        assert(False)
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        assert False, "Do not consider secure_mode"
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )  # generate, but throw away first generated Gaussian sample
        sum = zeros
        for _ in range(4):
            sum += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return sum / 2
    else:
        if config.is_debugging and config.debugging_type in ["without_noise", "without_noise_clipping"]:
            return torch.zeros(reference.shape, device=reference.device)
        elif config.is_debugging and config.debugging_type == "one_as_noise":
            return torch.ones(reference.shape, device=reference.device)
        elif config.is_debugging:
            assert False
        else:
            # TODO: suppose that only parameters of embedding layers are in CPU DRAM
            if reference.device == torch.device("cpu") and config.noise_base_optimize != "baseline":
                if config.noise_base_optimize == "multi_thread":
                    return custom_api_cpp.normal_multi_thread(std, reference.shape[0], reference.shape[1], config.noise_base_nthreads)
                else:
                    assert False
            else:
                return torch.normal(
                    mean=0,
                    std=std,
                    size=reference.shape,
                    device=reference.device,
                    generator=generator,
                )


class DPOptimizer(Optimizer):
    """
    ``torch.optim.Optimizer`` wrapper that adds additional functionality to clip per
    sample gradients and add Gaussian noise.

    Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
    ``DPOptimzer`` assumes that parameters over which it performs optimization belong
    to GradSampleModule and therefore have the ``grad_sample`` attribute.

    On a high level ``DPOptimizer``'s step looks like this:
    1) Aggregate ``p.grad_sample`` over all parameters to calculate per sample norms
    2) Clip ``p.grad_sample`` so that per sample norm is not above threshold
    3) Aggregate clipped per sample gradients into ``p.grad``
    4) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
    max grad norm limit (``std = noise_multiplier * max_grad_norm``).
    5) Call underlying optimizer to perform optimization step

    Examples:
        >>> module = MyCustomModel()
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> dp_optimizer = DPOptimizer(
        ...     optimizer=optimizer,
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ...     expected_batch_size=4,
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        module: GradSampleModule,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
        """

        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier
            max_grad_norm: max grad norm used for gradient clipping
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required is ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
        """
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(f"Unexpected value for loss_reduction: {loss_reduction}")

        if loss_reduction == "mean" and expected_batch_size is None:
            raise ValueError(
                "You must provide expected batch size of the loss reduction is mean"
            )

        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.loss_reduction = loss_reduction
        self.expected_batch_size = expected_batch_size
        self.step_hook = None
        self.generator = generator
        self.secure_mode = secure_mode

        self.param_groups = self.original_optimizer.param_groups
        self.defaults = self.original_optimizer.defaults
        self.state = self.original_optimizer.state
        self._step_skip_queue = []
        self._is_last_step_skipped = False

        for p in self.params:
            p.summed_grad = None
        
        self.module = module

        if config.dpsgd_mode == MODE_LAZYDP:
            self.cnt_iter = 0
            self.HT = list(torch.arange(len(self.module.emb_l)))
            for i in range(len(self.module.emb_l)):
                self.HT[i] = torch.zeros(self.module.emb_l[i].weight.shape[0], dtype=torch.int)
            self.stds_for_delayed_noise = list(torch.arange(len(self.module.emb_l)))
            self.lS_i_nxt = list(torch.arange(len(self.module.emb_l)))
            
                        

    def _get_flat_grad_sample(self, p: torch.Tensor):
        """
        Return parameter's per sample gradients as a single tensor.

        By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
        batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
        only one batch, and a list of tensors if gradients are accumulated over multiple
        steps. This is done to provide visibility into which sample belongs to which batch,
        and how many batches have been processed.

        This method returns per sample gradients as a single concatenated tensor, regardless
        of how many batches have been accumulated

        Args:
            p: Parameter tensor. Must have ``grad_sample`` attribute

        Returns:
            ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
            concatenating every tensor in ``p.grad_sample`` if it's a list

        Raises:
            ValueError
                If ``p`` is missing ``grad_sample`` attribute
        """

        if not hasattr(p, "grad_sample"):
            raise ValueError(
                "Per sample gradient not found. Are you using GradSampleModule?"
            )
        if p.grad_sample is None:
            raise ValueError(
                "Per sample gradient is not initialized. Not updated in backward pass?"
            )
        if isinstance(p.grad_sample, torch.Tensor):
            ret = p.grad_sample
        elif isinstance(p.grad_sample, list):
            ret = torch.cat(p.grad_sample, dim=0)
        else:
            raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        return ret

    def signal_skip_step(self, do_skip=True):
        """
        Signals the optimizer to skip an optimization step and only perform clipping and
        per sample gradient accumulation.

        On every call of ``.step()`` optimizer will check the queue of skipped step
        signals. If non-empty and the latest flag is ``True``, optimizer will call
        ``self.clip_and_accumulate``, but won't proceed to adding noise and performing
        the actual optimization step.
        It also affects the behaviour of ``zero_grad()``. If the last step was skipped,
        optimizer will clear per sample gradients accumulated by
        ``self.clip_and_accumulate`` (``p.grad_sample``), but won't touch aggregated
        clipped gradients (``p.summed_grad``)

        Used by :class:`~opacus.utils.batch_memory_manager.BatchMemoryManager` to
        simulate large virtual batches with limited memory footprint.

        Args:
            do_skip: flag if next step should be skipped
        """
        self._step_skip_queue.append(do_skip)

    def _check_skip_next_step(self, pop_next=True):
        """
        Checks if next step should be skipped by the optimizer.
        This is for large Poisson batches that get split into smaller physical batches
        to fit on the device. Batches that do not correspond to the end of a Poisson
        batch or thus `skipped` as their gradient gets accumulated for one big step.
        """
        if self._step_skip_queue:
            if pop_next:
                return self._step_skip_queue.pop(0)
            else:
                return self._step_skip_queue[0]
        else:
            return False

    @property
    def params(self) -> List[nn.Parameter]:
        """
        Returns a flat list of ``nn.Parameter`` managed by the optimizer
        """
        return params(self)

    @property
    def grad_samples(self) -> List[torch.Tensor]:
        """
        Returns a flat list of per sample gradient tensors (one per parameter)
        """
        ret = []
        for p in self.params:
            ret.append(self._get_flat_grad_sample(p))
        return ret

    @property
    def accumulated_iterations(self) -> int:
        """
        Returns number of batches currently accumulated and not yet processed.

        In other words ``accumulated_iterations`` tracks the number of forward/backward
        passed done in between two optimizer steps. The value would typically be 1,
        but there are possible exceptions.

        Used by privacy accountants to calculate real sampling rate.
        """
        return 1 # No exceptions in this research

        vals = []
        for p in self.params:
            if not hasattr(p, "grad_sample"):
                raise ValueError(
                    "Per sample gradient not found. Are you using GradSampleModule?"
                )
            if isinstance(p.grad_sample, torch.Tensor):
                vals.append(1)
            elif isinstance(p.grad_sample, list):
                vals.append(len(p.grad_sample))
            else:
                raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        if len(set(vals)) > 1:
            raise ValueError(
                "Number of accumulated steps is inconsistent across parameters"
            )
        return vals[0]

    def attach_step_hook(self, fn: Callable[[DPOptimizer], None]):
        """
        Attaches a hook to be executed after gradient clipping/noising, but before the
        actual optimization step.

        Most commonly used for privacy accounting.

        Args:
            fn: hook function. Expected signature: ``foo(optim: DPOptimizer)``
        """

        self.step_hook = fn

    def clip_and_accumulate(self, losses: torch.Tensor = None):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """
        if config.dpsgd_mode == MODE_DPSGD_B:
            assert losses == None
            if len(self.grad_samples[0]) == 0:
                # Empty batch
                assert False, "Exclude this exception"
                per_sample_clip_factor = torch.zeros((0,))
            else:
                config.profiler.start("Update_per_sample_clip_factor")
                per_param_norms = [
                    g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
                ]
                for i in range(len(per_param_norms)):
                    if(per_param_norms[i].device != config.device):
                        per_param_norms[i] = per_param_norms[i].to(config.device)
                # TODO: wrong implementation:
                # Even if num_gathers = 10, some sample might access 9 or 8 indices..
                # Current implementation of first backward just make (num_gathers) copies of output gradients of embeding layer
                # That means, the norm is over-valued
                per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
                per_sample_clip_factor = (
                    self.max_grad_norm / (per_sample_norms + 1e-6)
                ).clamp(max=1.0)
                config.profiler.end("Update_per_sample_clip_factor")
                
            if config.is_debugging and config.debugging_type == "without_noise_clipping":
                per_sample_clip_factor = torch.ones_like(per_sample_clip_factor)
            
            config.profiler.start("Update_clip_and_reduce")
            for p in self.params:
                _check_processed_flag(p.grad_sample)
                grad_sample = self._get_flat_grad_sample(p)
                
                if(not hasattr(p, 'inputs')):
                    config.profiler.start_l2("clip")
                    grad = contract("i,i...", per_sample_clip_factor, grad_sample)
                    config.profiler.end_l2("clip")
                else: # when embedding
                    config.profiler.start_l2("clip")
                    index, tmp_emb, offset = p.inputs
                    grad_sample = torch.einsum('i, ijk -> ijk', per_sample_clip_factor.to(torch.device("cpu")), grad_sample)
                    config.profiler.end_l2("clip")
                    
                    config.profiler.start_l2("coalesce")
                    if config.coalesce_optimize == "baseline":
                        grad = torch.sparse_coo_tensor(index.view(1, -1), grad_sample.view(-1, grad_sample.shape[-1]), p.shape).coalesce()
                    elif config.coalesce_optimize == "multi_thread_openmp":
                        grad = custom_api_cpp.coalesce_multi_thread_openmp(torch.sparse_coo_tensor(index.view(1, -1), grad_sample.view(-1, grad_sample.shape[-1]), p.shape), config.coalesce_nthreads)
                    elif config.coalesce_optimize == "multi_thread_embeddingbag":
                        grad = custom_api_cpp.coalesce_multi_thread_embeddingbag(torch.sparse_coo_tensor(index.view(1, -1), grad_sample.view(-1, grad_sample.shape[-1]), p.shape), config.coalesce_nthreads)
                    else:
                        assert False
                    config.profiler.end_l2("coalesce")

                config.profiler.start_l2("grad_to_summedgrad")
                if p.summed_grad is not None:
                    p.summed_grad += grad
                    assert False, "Exclude RNNs"
                else:
                    p.summed_grad = grad
                config.profiler.end_l2("grad_to_summedgrad")
                _mark_as_processed(p.grad_sample)
                
            config.profiler.end("Update_clip_and_reduce")
        elif config.dpsgd_mode in [MODE_DPSGD_R, MODE_DPSGD_F, MODE_LAZYDP, MODE_EANA]:
            config.profiler.start("clipping_factor")
            assert losses != None
            per_param_norms = []
            for p in self.params:
                if p.device != config.device:
                    assert len(p.grad_sample_norms) == 1
                    per_param_norms += [p.grad_sample_norms[0].to(config.device)]
                else:
                    per_param_norms += p.grad_sample_norms
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
                max=1.0
            )
            config.profiler.end("clipping_factor")
            
            if config.is_debugging and config.debugging_type == "without_noise_clipping":
                per_sample_clip_factor = torch.ones_like(per_sample_clip_factor)
                
            config.profiler.start("loss_clipping")
            # Use sum(), not mean(), to derive gradients in summed manner across mini-batch
            # This is to match with implementation of DP-SGD (B), summed gradients are added
            # with noise first, and then averaged 
            loss = (losses.reshape(1, -1)*per_sample_clip_factor).sum()
            config.profiler.end("loss_clipping")
            
            config.profiler.start("2nd_backprop")
            config.profiler.start_l2("backward")
            self.module.disable_hooks()
            loss.backward()
            config.profiler.end_l2("backward")

            # In LazyDP, do coalescing after merging (sparse) gradient and (sparse) noise
            if config.dpsgd_mode != MODE_LAZYDP:
                config.profiler.start_l2("coalesce")
                if config.coalesce_optimize == "baseline":
                    for param in self.module.emb_l.parameters():
                        param.grad = param.grad.coalesce()
                elif config.coalesce_optimize == "multi_thread_openmp":
                    for param in self.module.emb_l.parameters():
                        param.grad = custom_api_cpp.coalesce_multi_thread_openmp(param.grad, config.coalesce_nthreads)
                elif config.coalesce_optimize == "multi_thread_embeddingbag":
                    for param in self.module.emb_l.parameters():
                        param.grad = custom_api_cpp.coalesce_multi_thread_embeddingbag(param.grad, config.coalesce_nthreads)
                config.profiler.end_l2("coalesce")

            self.module.enable_hooks()
            config.profiler.end("2nd_backprop")

            config.profiler.start("summedgrad_to_grad")
            for p in self.params:
                p.summed_grad = p.grad
                p.grad = None
            config.profiler.end("summedgrad_to_grad")
        else:
            assert False, "Invalid mode of DP-SGD"

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        config.profiler.start("Update_noise")
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            # TODO: suppose that only parameters of embedding layers are in CPU DRAM
            if p.device == torch.device('cpu') and (config.dpsgd_mode in [MODE_LAZYDP, MODE_EANA]): # emgedding layer
                if config.dpsgd_mode == MODE_EANA: # when MODE_EANA
                    config.profiler.start_l2("generate_noise_emb")
                    noise = _generate_noise(
                        std=self.noise_multiplier * self.max_grad_norm,
                        reference=p.summed_grad.values(),
                        generator=self.generator,
                        secure_mode=self.secure_mode,
                    )
                    config.profiler.end_l2("generate_noise_emb")
                    
                    config.profiler.start_l2("add_noise_emb")
                    p.summed_grad.values().add_(noise)
                    p.grad = p.summed_grad
                    config.profiler.end_l2("add_noise_emb")
                else: # when MODE_LAZYDP
                    config.profiler.start_l2("bypass_emb")
                    p.grad = p.summed_grad
                    config.profiler.end_l2("bypass_emb")
            else: #  for MLP layers or other DP-SGD algorithms (DP-SGD(B, R))
                if p.device == torch.device('cpu'):
                    config.profiler.start_l2("generate_noise_emb")
                else:
                    config.profiler.start_l2("generate_noise_mlp")

                noise = _generate_noise(
                    std=self.noise_multiplier * self.max_grad_norm,
                    reference=p.summed_grad,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                ) 
                
                if p.device == torch.device('cpu'):
                    config.profiler.end_l2("generate_noise_emb")
                    config.profiler.start_l2("add_noise_emb")
                else:
                    config.profiler.end_l2("generate_noise_mlp")
                    config.profiler.start_l2("add_noise_mlp")
                    
                if p.device == torch.device('cpu') and (config.dpsgd_mode in [MODE_DPSGD_B, MODE_DPSGD_R, MODE_DPSGD_F]):
                    noise[p.summed_grad.indices().view(-1)] += p.summed_grad.values()
                    p.grad = noise
                else:
                    p.grad = noise.add_(p.summed_grad)
                
                if p.device == torch.device('cpu'):
                    config.profiler.end_l2("add_noise_emb")
                else:
                    config.profiler.end_l2("add_noise_mlp")

            _mark_as_processed(p.summed_grad)
        config.profiler.end("Update_noise")
        

    def scale_grad(self):
        """
        Applies given ``loss_reduction`` to ``p.grad``.

        Does nothing if ``loss_reduction="sum"``. Divides gradients by
        ``self.expected_batch_size`` if ``loss_reduction="mean"``
        """
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= self.expected_batch_size * self.accumulated_iterations

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.

        Clears ``p.grad``, ``p.grad_sample`` and ``p.summed_grad`` for all of it's parameters

        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` and
            ``p.summed_grad`` is never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list

        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """

        set_to_none = True # Always set to none
        if set_to_none is False:
            logger.debug(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample and p.summed_grad to None due to "
                "non-trivial gradient accumulation behaviour"
            )
        for p in self.params:
            if config.dpsgd_mode == MODE_DPSGD_B:
                p.grad_sample = None

            if not self._is_last_step_skipped:
                p.summed_grad = None

        self.original_optimizer.zero_grad(set_to_none)

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None, losses: torch.Tensor = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.clip_and_accumulate(losses)
        
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False
        
        self.add_noise()

        if config.dpsgd_mode == MODE_LAZYDP:
            config.profiler.start("Update_set_emb_to_noise_update")
            self.set_emb_to_noise_update()
            config.profiler.end("Update_set_emb_to_noise_update")
            
            config.profiler.start("Update_delayed_noise_update")
            self.do_delayed_noise_update()
            config.profiler.end("Update_delayed_noise_update")
        

        # TODO: This can be merged in learning rate
        # New learining rate = original learning rate / batch size
        if config.is_debugging:
            self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, closure: Optional[Callable[[], float]] = None, losses: torch.Tensor = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step(losses=losses):
            config.profiler.start("Update_original")
            ret = self.original_optimizer.step()
            config.profiler.end("Update_original")
            return ret
        else:
            return None

    def __repr__(self):
        return self.original_optimizer.__repr__()

    def state_dict(self):
        return self.original_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.original_optimizer.load_state_dict(state_dict)

    def set_emb_to_noise_update(self):
        if self.lS_i_nxt == None:
            return
        
        lS_i_nxt = self.lS_i_nxt

        for i in range(len(lS_i_nxt)):
            self.stds_for_delayed_noise[i] = ((self.cnt_iter - self.HT[i][self.lS_i_nxt[i]])**(1/2))*self.noise_multiplier*self.max_grad_norm
                
    def do_delayed_noise_update(self):
        dim = self.module.emb_l[0].weight.shape[1]
        for i in range(len(self.module.emb_l)):
            if self.lS_i_nxt != None:
                config.profiler.start_l2("generate_noise_emb")
                std = self.stds_for_delayed_noise[i]
                if config.is_debugging and config.debugging_type in ["without_noise", "without_nosie_clipping"]:
                    delays = (std/self.noise_multiplier/self.max_grad_norm)**2
                    v = torch.cat([torch.zeros(std.shape[0], dim) * delays.unsqueeze(1), torch.zeros(config.cur_batch_size * config.num_gathers_list[i], dim)])
                elif config.is_debugging and config.debugging_type == "one_as_noise":
                    delays = (std/self.noise_multiplier/self.max_grad_norm)**2
                    v = torch.cat([torch.ones(std.shape[0], dim) * delays.unsqueeze(1), torch.zeros(config.cur_batch_size * config.num_gathers_list[i], dim)])
                elif config.is_debugging:
                    assert False
                else:
                    v = custom_api_cpp.normal_multi_thread_with_extra(std, dim, config.cur_batch_size * config.num_gathers_list[i], config.noise_final_nthreads)
                config.profiler.end_l2("generate_noise_emb")
                
                config.profiler.start_l2("add_noise_emb")
                sparse_grad = self.params[i].grad
                n_rows_noise = std.shape[0]
                v[n_rows_noise:] = sparse_grad._values()
                new_indices = torch.empty((1, v.shape[0]), dtype=torch.int64)
                new_indices[0][:n_rows_noise] = self.lS_i_nxt[i]
                new_indices[0][n_rows_noise:] = sparse_grad._indices()[0]
                n_rows_total = self.params[i].shape[0]
                noisy_grad = torch.sparse_coo_tensor(new_indices, v, (n_rows_total, dim))
                config.profiler.end_l2("add_noise_emb")

                assert config.cur_batch_size * config.num_gathers_list[i] == sparse_grad._indices().shape[1]
                assert v.shape[0] == config.cur_batch_size * config.num_gathers_list[i] + n_rows_noise
                assert v.shape[1] == dim
            else:
                noisy_grad = self.params[i].grad
                
            config.profiler.start_l2("coalesce")
            if config.coalesce_optimize == "baseline":
                self.params[i].grad = noisy_grad.coalesce()
            elif config.coalesce_optimize == "multi_thread_openmp":
                self.params[i].grad = custom_api_cpp.coalesce_multi_thread_openmp(noisy_grad, config.coalesce_nthreads)
            elif config.coalesce_optimize == "multi_thread_embeddingbag":
                self.params[i].grad = custom_api_cpp.coalesce_multi_thread_embeddingbag(noisy_grad, config.coalesce_nthreads)
            else:
                assert False
            config.profiler.end_l2("coalesce")
            
    def set_HT_increase_cnt_iter(self):
        lS_i_nxt = self.lS_i_nxt
        assert len(lS_i_nxt) == len(self.module.emb_l)
        for i in range(len(lS_i_nxt)):
            self.HT[i][lS_i_nxt[i]] = self.cnt_iter
        self.cnt_iter += 1

    def set_lS_i(self, lS_i_nxt):
        if lS_i_nxt == None:
            self.lS_i_nxt = None
            return
        
        for i in range(len(lS_i_nxt)):
            if config.unique_optimize == "baseline":
                self.lS_i_nxt[i] = lS_i_nxt[i].unique()
            elif config.unique_optimize == "multi_thread":
                self.lS_i_nxt[i] = custom_api_cpp.unique_multi_thread(lS_i_nxt[i])
            else:
                assert False
        assert True
        
    def add_remaining_noise_for_debugging(self):
        assert config.is_debugging == True
        
        if config.debugging_type in ["without_noise", "without_noise_clipping"]:
            return
        
        assert config.debugging_type == "one_as_noise"
        
        with torch.no_grad():
            for i in range(len(self.module.emb_l)):
                remaining_noise = torch.ones_like(self.module.emb_l[i].weight) * (self.cnt_iter - self.HT[i]).unsqueeze(1)
                self.module.emb_l[i].weight.add_(remaining_noise, alpha=-self.original_optimizer.param_groups[0]["lr"]/(self.expected_batch_size * self.accumulated_iterations)) # self.expected_batch_size * self.accumulated_iterations : same as "scale_grad()"