import unittest

import torch
from opacus import PrivacyEngine
from opacus.grad_sample.gsm_exp_weights import API_CUTOFF_VERSION
from torch.utils.data import DataLoader

from .utils import (
    BasicSupportedModule,
    CustomLinearModule,
    LinearWithExtraParam,
    MatmulModule,
)


class PrivacyEngineValidationTest(unittest.TestCase):
    """
    This test case checks end-to-end model validation performed in `.make_private`
    method. It covers performed in `ModuleValidator`, `GradSampleModule`, as well as
    their interplay
    """

    def setUp(self) -> None:
        self.privacy_engine = PrivacyEngine()

    def _init(self, module, size, batch_size=10):
        optim = torch.optim.SGD(module.parameters(), lr=0.1)
        dl = DataLoader(
            dataset=[torch.randn(*size) for _ in range(100)],
            batch_size=batch_size,
        )

        return module, optim, dl

    def test_supported_hooks(self):
        module, optim, dl = self._init(BasicSupportedModule(), size=(16, 5))

        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="hooks",
        )

        for x in dl:
            module(x)

    @unittest.skipIf(
        torch.__version__ < API_CUTOFF_VERSION, "not supported in this torch version"
    )
    def test_supported_ew(self):
        module, optim, dl = self._init(BasicSupportedModule(), size=(16, 5))

        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="ew",
        )

        for x in dl:
            module(x)

    def test_custom_linear_hooks(self):
        module, optim, dl = self._init(CustomLinearModule(5, 8), size=(16, 5))
        try:
            gsm, _, _ = self.privacy_engine.make_private(
                module=module,
                optimizer=optim,
                data_loader=dl,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                grad_sample_mode="hooks",
            )
            self.assertTrue(hasattr(gsm._module, "ft_compute_sample_grad"))
        except ImportError:
            print("Test not ran because functorch not imported")

    @unittest.skipIf(
        torch.__version__ < API_CUTOFF_VERSION, "not supported in this torch version"
    )
    def test_custom_linear_ew(self):
        module, optim, dl = self._init(CustomLinearModule(5, 8), size=(16, 5))

        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="ew",
        )

        for x in dl:
            module(x)

    def test_unsupported_hooks(self):
        try:
            module, optim, dl = self._init(MatmulModule(5, 8), size=(16, 5))

            gsm, _, _ = self.privacy_engine.make_private(
                module=module,
                optimizer=optim,
                data_loader=dl,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                grad_sample_mode="hooks",
            )
            self.assertTrue(hasattr(gsm._module, "ft_compute_sample_grad"))
        except ImportError:
            print("Test not ran because functorch not imported")

    @unittest.skipIf(
        torch.__version__ < API_CUTOFF_VERSION, "not supported in this torch version"
    )
    def test_unsupported_ew(self):
        module, optim, dl = self._init(
            MatmulModule(input_features=5, output_features=10),
            size=(16, 5),
            batch_size=12,
        )

        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="ew",
        )

        with self.assertRaises(RuntimeError):
            for x in dl:
                module(x)

    def test_extra_param_hooks_requires_grad(self):
        module, optim, dl = self._init(LinearWithExtraParam(5, 8), size=(16, 5))
        try:
            gsm, _, _ = self.privacy_engine.make_private(
                module=module,
                optimizer=optim,
                data_loader=dl,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                grad_sample_mode="hooks",
            )
            self.assertTrue(hasattr(gsm._module, "ft_compute_sample_grad"))
            gsm._close()
        except ImportError:
            print("Test not ran because functorch not imported")

    def test_extra_param_hooks_no_requires_grad(self):
        module, optim, dl = self._init(LinearWithExtraParam(5, 8), size=(16, 5))
        module.extra_param.requires_grad = False
        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="hooks",
        )

        for x in dl:
            module(x)

    @unittest.skipIf(
        torch.__version__ < API_CUTOFF_VERSION, "not supported in this torch version"
    )
    def test_extra_param_ew(self):
        module, optim, dl = self._init(LinearWithExtraParam(5, 8), size=(16, 5))
        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="ew",
        )
        with self.assertRaises(RuntimeError):
            for x in dl:
                module(x)

    @unittest.skipIf(
        torch.__version__ < API_CUTOFF_VERSION, "not supported in this torch version"
    )
    def test_extra_param_disabled_ew(self):
        module, optim, dl = self._init(LinearWithExtraParam(5, 8), size=(16, 5))
        module.extra_param.requires_grad = False

        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="ew",
        )

        for x in dl:
            module(x)
