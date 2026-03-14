import torch
import torch.nn.functional as F
from torch import nn

from scipy.linalg import hadamard

from ..utils import FPQuantConfig, FPQuantDtype, validate_config
from .linear_fns import (
    FPQuant4x16MasterFn,
    FPQuant4x4MasterFn,
    FPQuant4x8MasterFn,
    FPQuant4x8NoMasterFn,
    FPQuant4x16NoMasterFn,
    forward_quantize,
)
from .qutlass_ops import HAS_QUTLASS
from .pseudoquant_linear_fns import (
    PseudoQuant4x16MasterFn,
    PseudoQuant4x16NoMasterFn,
    forward_pseudoquantize,
)


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )


def get_identity_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.eye(group_size, dtype=dtype, device=device, requires_grad=False)


def get_gsr_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    hadamard_matrix = get_hadamard_matrix(group_size, dtype, device)
    sign_changes = torch.diff(hadamard_matrix, dim=0).ne(0).sum(dim=0)
    sorted_indices = torch.argsort(sign_changes)
    return hadamard_matrix[:, sorted_indices].contiguous()


class FPQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: FPQuantConfig,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        validate_config(config)

        if not HAS_QUTLASS and not config.pseudoquantization:
            raise ValueError(
                "QuTLASS is not installed. Can only run with `pseudoquantization=True` in the quantization config. If you have a Blackwell GPU, you can install QuTLASS from https://github.com/IST-DASLab/QuTLASS"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.dqweight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.config = config

        # Quantized tensors buffers
        if self.config.forward_dtype == FPQuantDtype.MXFP4:
            self.register_buffer(
                "qweight",
                torch.empty(
                    self.weight.shape[0],
                    self.weight.shape[1] // 2,
                    dtype=torch.uint8,
                    device=self.weight.device,
                ),
            )
            self.register_buffer(
                "scales",
                torch.empty(
                    self.weight.shape[0],
                    self.weight.shape[1] // 32,
                    dtype=torch.uint8,
                    device=self.weight.device,
                ),
            )
        elif self.config.forward_dtype == FPQuantDtype.NVFP4:
            self.register_buffer(
                "qweight",
                torch.empty(
                    self.weight.shape[0],
                    self.weight.shape[1] // 2,
                    dtype=torch.uint8,
                    device=self.weight.device,
                ),
            )
            self.register_buffer(
                "scales",
                torch.empty(
                    self.weight.shape[0],
                    self.weight.shape[1] // 16,
                    dtype=torch.uint8,
                    device=self.weight.device,
                ),
            )
        else:
            raise ValueError(f"Unsupported forward dtype: {config.forward_dtype}")

        # Global scale buffers
        self.register_buffer(
            "weight_global_scale",
            torch.empty(
                1,
                **factory_kwargs,
            ),
        )
        self.register_buffer(
            "act_global_scale",
            torch.empty(
                1,
                **factory_kwargs,
            ),
        )

        # Rotation matrices buffers
        self.register_buffer(
            "forward_hadamard_matrix",
            torch.empty(
                self.config.hadamard_group_size,
                self.config.hadamard_group_size,
                **factory_kwargs,
            ),
        )
        self.register_buffer(
            "backward_hadamard_matrix",
            torch.empty(
                self.config.hadamard_group_size,
                self.config.hadamard_group_size,
                **factory_kwargs,
            ),
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Original code assumed [hadamard_group_size, hadamard_group_size] buffers.
        # New: allow loading exported checkpoints that store per-group transform banks
        # (e.g. [num_groups, group_size, group_size]) by resizing placeholder buffers.
        for buffer_name in ("forward_hadamard_matrix", "backward_hadamard_matrix"):
            key = prefix + buffer_name
            if key in state_dict:
                incoming = state_dict[key]
                current = self._buffers.get(buffer_name, None)
                if current is None or tuple(current.shape) != tuple(incoming.shape):
                    self._buffers[buffer_name] = torch.empty_like(incoming)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


    @torch.no_grad()
    def pre_forward(self):
        # ===== [新增] 1) 如果是导出的 pseudoquant checkpoint：dqweight 已经�?state_dict �?=====
        # 这种情况下不需要（也不允许）再�?master weight 触发 pseudoquant 过程�?        # 否则会在 CPU 上触�?Triton 或覆盖已加载�?dqweight�?        
        if self.config.pseudoquantization and (not self.config.store_master_weights):
            if getattr(self, "dqweight", None) is not None:
                # 确保不保�?不依�?master weight 路径
                self.weight = None
                self.qweight = None
                self.scales = None
                setattr(self, "_fpq_deferred_pre_forward", False)
                return

        # ===== [新增] 2) 若需要从 weight 生成量化参数，但当前不在 CUDA/XPU，则延迟 =====
        # 注意：pseudoquant �?Triton kernel 只能�?CUDA/XPU 上跑�?        
        if getattr(self, "weight", None) is None:
            # 这里一般只会出现在不完�?不匹配的 checkpoint；先标记延迟，forward 时再报更明确的错
            setattr(self, "_fpq_deferred_pre_forward", True)
            return

        weight_in_device = self.weight.data.device.type in ["cuda", "xpu"]
        if not weight_in_device:
            setattr(self, "_fpq_deferred_pre_forward", True)
            return

        setattr(self, "_fpq_deferred_pre_forward", False)

        # ===== 下面保持你原来的逻辑不变 =====
        assert (
            self.weight.shape[1] % self.config.hadamard_group_size == 0
        ), f"Weight shape must be divisible by hadamard group size: {self.weight.shape[1]} % {self.config.hadamard_group_size} = {self.weight.shape[1] % self.config.hadamard_group_size}"

        if not self.config.pseudoquantization:
            assert (
                weight_in_device
            ), f"Weight must be on CUDA or XPU, but is on {self.weight.device}"

        if self.config.transform_init == "hadamard":
            transform_init_fn = get_hadamard_matrix
        elif self.config.transform_init == "identity":
            transform_init_fn = get_identity_matrix
        elif self.config.transform_init == "gsr":
            transform_init_fn = get_gsr_matrix
        else:
            raise ValueError(f"Invalid transform init: {self.config.transform_init}")

        self.forward_hadamard_matrix = nn.Buffer(
            transform_init_fn(
                self.config.hadamard_group_size,
                self.weight.dtype,
                self.weight.device,
            ),
        )
        self.backward_hadamard_matrix = nn.Buffer(
            transform_init_fn(
                self.config.hadamard_group_size,
                self.weight.dtype,
                self.weight.device,
            ),
        )

        if (
            self.config.forward_dtype == FPQuantDtype.MXFP4
            and self.config.forward_method == "quest"
        ):
            global_scale_val = 1.0
        elif self.config.forward_method == "abs_max":
            global_scale_val = 3.0
        elif self.config.forward_dtype == FPQuantDtype.NVFP4:
            global_scale_val = 10.0

        self.weight_global_scale = nn.Buffer(
            torch.tensor(
                [global_scale_val],
                dtype=self.weight.dtype,
                device=self.weight.device,
                requires_grad=False,
            ),
        )
        self.act_global_scale = nn.Buffer(
            torch.tensor(
                [global_scale_val],
                dtype=self.weight.dtype,
                device=self.weight.device,
                requires_grad=False,
            ),
        )

        if self.config.store_master_weights:
            self.qweight = None
            self.scales = None
            self.dqweight = None
        elif self.config.pseudoquantization:
            weight_dq, _ = forward_pseudoquantize(
                self.weight.data,
                self.forward_hadamard_matrix,
                self.weight_global_scale,
                self.config.forward_dtype,
                self.config.forward_method,
            )
            self.dqweight = nn.Parameter(weight_dq, requires_grad=False)
            self.weight = None
            self.qweight = None
            self.scales = None
        else:
            weight_q, scales, _ = forward_quantize(
                self.weight,
                self.forward_hadamard_matrix,
                self.weight_global_scale,
                self.config.forward_dtype,
                self.config.forward_method,
            )
            self.qweight = nn.Parameter(weight_q, requires_grad=False)
            self.scales = nn.Parameter(
                scales.view(dtype=torch.uint8), requires_grad=False
            )
            self.weight = None
            self.dqweight = None


    def forward(self, x) -> torch.Tensor:
        # Deferred pre_forward is finalized once activations are on CUDA/XPU.
        if getattr(self, "_fpq_deferred_pre_forward", False):
            if x.device.type in ["cuda", "xpu"]:
                self.pre_forward()
            # If still deferred, raise a clear device-placement error.
            if getattr(self, "_fpq_deferred_pre_forward", False):
                dev = None
                if getattr(self, "weight", None) is not None:
                    dev = self.weight.device
                elif getattr(self, "dqweight", None) is not None:
                    dev = self.dqweight.device
                raise ValueError(
                    f"FPQuantLinear pre_forward() is deferred because weights are not on CUDA/XPU "
                    f"(current device: {dev}). Ensure the model is placed on GPU via device_map "
                    f"and do not offload FPQuant modules to CPU."
                )

        # ===== 下面保持你原来的分支不变 =====
        if (
            self.config.forward_dtype == FPQuantDtype.MXFP4
            and self.config.backward_dtype == FPQuantDtype.MXFP4
            and self.config.store_master_weights == True
            and self.config.pseudoquantization == False
        ):
            return FPQuant4x4MasterFn.apply(
                x,
                self.weight,
                self.weight_global_scale,
                self.act_global_scale,
                self.bias,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
        elif (
            self.config.forward_dtype == FPQuantDtype.MXFP4
            and self.config.backward_dtype == FPQuantDtype.MXFP8
            and self.config.store_master_weights == True
            and self.config.pseudoquantization == False
        ):
            return FPQuant4x8MasterFn.apply(
                x,
                self.weight,
                self.weight_global_scale,
                self.act_global_scale,
                self.bias,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
        elif (
            self.config.forward_dtype == FPQuantDtype.MXFP4
            and self.config.backward_dtype == FPQuantDtype.MXFP8
            and self.config.store_master_weights == False
            and self.config.pseudoquantization == False
        ):
            return FPQuant4x8NoMasterFn.apply(
                x,
                self.qweight,
                self.scales,
                self.weight_global_scale,
                self.act_global_scale,
                self.bias,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
        elif (
            self.config.forward_dtype in (FPQuantDtype.MXFP4, FPQuantDtype.NVFP4)
            and self.config.backward_dtype == FPQuantDtype.BF16
            and self.config.store_master_weights == True
            and self.config.pseudoquantization == False
        ):
            return FPQuant4x16MasterFn.apply(
                x,
                self.weight,
                self.weight_global_scale,
                self.act_global_scale,
                self.bias,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
        elif (
            self.config.forward_dtype in (FPQuantDtype.MXFP4, FPQuantDtype.NVFP4)
            and self.config.backward_dtype == FPQuantDtype.BF16
            and self.config.store_master_weights == False
            and self.config.pseudoquantization == False
        ):
            return FPQuant4x16NoMasterFn.apply(
                x,
                self.qweight,
                self.scales,
                self.weight_global_scale,
                self.act_global_scale,
                self.bias,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
        elif (
            self.config.forward_dtype in (FPQuantDtype.MXFP4, FPQuantDtype.NVFP4)
            and self.config.backward_dtype == FPQuantDtype.BF16
            and self.config.store_master_weights == True
            and self.config.pseudoquantization == True
        ):
            return PseudoQuant4x16MasterFn.apply(
                x,
                self.weight,
                self.weight_global_scale,
                self.act_global_scale,
                self.bias,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
        elif (
            self.config.forward_dtype in (FPQuantDtype.MXFP4, FPQuantDtype.NVFP4)
            and self.config.backward_dtype == FPQuantDtype.BF16
            and self.config.store_master_weights == False
            and self.config.pseudoquantization == True
        ):
            return PseudoQuant4x16NoMasterFn.apply(
                x,
                self.dqweight,
                self.weight_global_scale,
                self.act_global_scale,
                self.bias,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
        else:
            raise ValueError(
                f"Forward dtype: {self.config.forward_dtype}, backward dtype: {self.config.backward_dtype}, "
                f"store_master_weights: {self.config.store_master_weights}, pseudoquantization: {self.config.pseudoquantization} isn't supported yet."
            )



