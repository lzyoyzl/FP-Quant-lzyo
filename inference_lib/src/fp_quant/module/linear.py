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


        # New: track pseudoquant checkpoint integrity across sharded loading.
        # HF may stream state_dict in multiple passes, so we cannot validate all keys in one call.
        self._fpq_seen_state_keys = set()
        self._fpq_runtime_state_validated = False
        self._fpq_loaded_from_state_dict = False

    def _validate_transform_matrix_shape(
        self,
        tensor: torch.Tensor,
        name: str,
    ) -> list[str]:
        errors = []
        if tensor.ndim == 2:
            if tensor.shape[0] != tensor.shape[1]:
                errors.append(
                    f"{name} must be square when 2D, got shape={tuple(tensor.shape)}."
                )
            elif self.in_features % tensor.shape[0] != 0:
                errors.append(
                    f"{name} 2D size {tensor.shape[0]} must divide in_features={self.in_features}."
                )
        elif tensor.ndim == 3:
            num_groups, group_size_r, group_size_c = tensor.shape
            if group_size_r != group_size_c:
                errors.append(
                    f"{name} group blocks must be square, got shape={tuple(tensor.shape)}."
                )
            elif num_groups * group_size_r != self.in_features:
                errors.append(
                    f"{name} shape={tuple(tensor.shape)} is incompatible with in_features={self.in_features}."
                )
        else:
            errors.append(
                f"{name} must be 2D or 3D, got ndim={tensor.ndim}, shape={tuple(tensor.shape)}."
            )
        return errors

    def _validate_pseudoquant_runtime_state(self) -> list[str]:
        errors = []
        required_keys = {
            "dqweight",
            "forward_hadamard_matrix",
            "backward_hadamard_matrix",
            "weight_global_scale",
            "act_global_scale",
        }

        # If at least one key has been loaded from checkpoint, require all pseudoquant keys.
        if len(self._fpq_seen_state_keys) > 0:
            missing_keys = sorted(required_keys - self._fpq_seen_state_keys)
            if missing_keys:
                errors.append(f"missing pseudoquant checkpoint keys: {missing_keys}")

        if self.dqweight is None:
            errors.append("dqweight is None for pseudoquant no-master path.")
        else:
            expected_shape = (self.out_features, self.in_features)
            if tuple(self.dqweight.shape) != expected_shape:
                errors.append(
                    f"dqweight shape mismatch: got {tuple(self.dqweight.shape)}, expected {expected_shape}."
                )
            elif not torch.isfinite(self.dqweight).all():
                errors.append("dqweight contains NaN/Inf.")

        errors.extend(
            self._validate_transform_matrix_shape(
                self.forward_hadamard_matrix, "forward_hadamard_matrix"
            )
        )
        errors.extend(
            self._validate_transform_matrix_shape(
                self.backward_hadamard_matrix, "backward_hadamard_matrix"
            )
        )

        if tuple(self.forward_hadamard_matrix.shape) != tuple(self.backward_hadamard_matrix.shape):
            errors.append(
                "forward_hadamard_matrix and backward_hadamard_matrix shape mismatch: "
                f"{tuple(self.forward_hadamard_matrix.shape)} vs {tuple(self.backward_hadamard_matrix.shape)}."
            )

        if not torch.isfinite(self.forward_hadamard_matrix).all():
            errors.append("forward_hadamard_matrix contains NaN/Inf.")
        if not torch.isfinite(self.backward_hadamard_matrix).all():
            errors.append("backward_hadamard_matrix contains NaN/Inf.")

        if self.weight_global_scale is None or self.weight_global_scale.numel() < 1:
            errors.append("weight_global_scale is missing or empty.")
        elif not torch.isfinite(self.weight_global_scale).all():
            errors.append("weight_global_scale contains NaN/Inf.")

        if self.act_global_scale is None or self.act_global_scale.numel() < 1:
            errors.append("act_global_scale is missing or empty.")
        elif not torch.isfinite(self.act_global_scale).all():
            errors.append("act_global_scale contains NaN/Inf.")

        return errors

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
        required_pseudoquant_keys = (
            "dqweight",
            "forward_hadamard_matrix",
            "backward_hadamard_matrix",
            "weight_global_scale",
            "act_global_scale",
        )
        for key_name in required_pseudoquant_keys:
            key = prefix + key_name
            if key in state_dict:
                self._fpq_seen_state_keys.add(key_name)

        for buffer_name in ("forward_hadamard_matrix", "backward_hadamard_matrix"):
            key = prefix + buffer_name
            if key in state_dict:
                incoming = state_dict[key]
                current = self._buffers.get(buffer_name, None)
                if current is None or tuple(current.shape) != tuple(incoming.shape):
                    self._buffers[buffer_name] = torch.empty_like(incoming)
                # New: validate incoming matrix shape early for clearer checkpoint errors.
                incoming_errors = self._validate_transform_matrix_shape(incoming, buffer_name)
                for err in incoming_errors:
                    error_msgs.append(f"{prefix}{err}")
                if not torch.isfinite(incoming).all():
                    error_msgs.append(f"{key} contains NaN/Inf in checkpoint.")

        dqweight_key = prefix + "dqweight"
        if dqweight_key in state_dict:
            incoming_dqweight = state_dict[dqweight_key]
            expected_shape = (self.out_features, self.in_features)
            if tuple(incoming_dqweight.shape) != expected_shape:
                error_msgs.append(
                    f"{dqweight_key} shape mismatch: got {tuple(incoming_dqweight.shape)}, expected {expected_shape}."
                )
            if not torch.isfinite(incoming_dqweight).all():
                error_msgs.append(f"{dqweight_key} contains NaN/Inf in checkpoint.")

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._fpq_loaded_from_state_dict = True


    @torch.no_grad()
    def pre_forward(self):
        # 1) For exported pseudoquant checkpoints (no-master), dqweight is already loaded.
        # Skip runtime pseudoquantization from master weights to avoid CPU/Triton issues
        # and to preserve the loaded dqweight tensors.
        if self.config.pseudoquantization and (not self.config.store_master_weights):
            if getattr(self, "dqweight", None) is not None:
                # Force no-master inference path.
                self.weight = None
                self.qweight = None
                self.scales = None
                setattr(self, "_fpq_deferred_pre_forward", False)
                return

        # 2) If quant params must be generated from self.weight but weight is not on CUDA/XPU,
        # defer pre_forward until first GPU activation arrives.
        if getattr(self, "weight", None) is None:
            # Usually indicates an incomplete or mismatched checkpoint; keep deferred flag
            # so forward() can raise a clearer error later.
            setattr(self, "_fpq_deferred_pre_forward", True)
            return

        weight_in_device = self.weight.data.device.type in ["cuda", "xpu"]
        if not weight_in_device:
            setattr(self, "_fpq_deferred_pre_forward", True)
            return

        setattr(self, "_fpq_deferred_pre_forward", False)

        # Keep original pre_forward logic below.
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
        # New: fail fast if pseudoquant no-master state is partially loaded / malformed.
        if (
            self.config.pseudoquantization
            and not self.config.store_master_weights
            and not self._fpq_runtime_state_validated
        ):
            runtime_errors = self._validate_pseudoquant_runtime_state()
            # For non-checkpoint path, only enforce pre_forward requirement when not deferred.
            if (
                len(self._fpq_seen_state_keys) == 0
                and self.weight is not None
                and self.dqweight is not None
                and not getattr(self, "_fpq_deferred_pre_forward", False)
            ):
                runtime_errors.append(
                    "pseudoquant no-master layer is not checkpoint-loaded and still has master weight; "
                    "call pre_forward() before inference or load a valid pseudoquant checkpoint."
                )
            # If pre_forward is intentionally deferred to first CUDA/XPU activation,
            # skip strict state validation until deferred pre_forward is finalized.
            if len(self._fpq_seen_state_keys) == 0 and getattr(self, "_fpq_deferred_pre_forward", False):
                runtime_errors = []
            if runtime_errors:
                raise ValueError(
                    "FPQuantLinear pseudoquant state validation failed: "
                    + " | ".join(runtime_errors)
                )
            self._fpq_runtime_state_validated = True

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
        # Keep original forward branches below.
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




