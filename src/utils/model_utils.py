import numpy as np
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import AutoConfig

from .common_utils import to
from .llama_utils import QuantizedLlamaMLP, QuantizedLlamaAttention
from .qwen3_utils import QuantizedQwen3MLP, QuantizedQwen3Attention

### Calibration utils and modules

LINEAR_LAYERS = (nn.Linear, _ConvNd)


class ForwardInterrupt(Exception):
    pass


class InputCollector(nn.Module):
    """
    Wrap a transformer block to collect the actual inputs passed to it,
    then interrupt forward to avoid running the whole model.

    Key compatibility feature:
    - Proxy unknown attribute accesses to the wrapped module (e.g. Qwen3 needs
      decoder_layer.attention_type before calling the layer).
    """

    def __init__(self, module: nn.Module, cpu_offload: bool = False):
        super().__init__()
        self.module = module
        self.cpu_offload = cpu_offload
        self.input_args = []
        self.input_kwargs = []

        # Qwen3 会在进入 layer forward 前读取 decoder_layer.attention_type
        # 这里显式镜像一份，避免某些路径下 __getattr__ 没被触发/被绕开
        if hasattr(module, "attention_type"):
            self.attention_type = getattr(module, "attention_type")

    def forward(self, *input_args, **input_kwargs):
        """
        Assumes that the wrapped module has a single
        input that can reside in inputs or input_kwargs.
        """
        if self.cpu_offload:
            input_args = to(input_args, device="cpu")
            input_kwargs = to(input_kwargs, device="cpu")

        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)

        # 中断：让上层 try/except ForwardInterrupt 捕获
        raise ForwardInterrupt

    def __getattr__(self, name: str):
        """
        先走 nn.Module 默认查找（参数/子模块/自身属性），找不到再透传到被包装的 module。
        这能修复 Qwen3 对 decoder_layer.attention_type 等属性的访问。
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            # 注意：这里必须取到 wrapped module，再把属性访问透传过去
            wrapped = super().__getattr__("module")
            return getattr(wrapped, name)

def get_number_of_rows_and_cols(layer):
    return layer.weight.shape[0], np.prod(layer.weight.shape[1:])

def get_mlp_layer(config: AutoConfig):
    if config.model_type == "llama":
        return QuantizedLlamaMLP
    elif config.model_type == "qwen3":
        return QuantizedQwen3MLP
    else:
        raise ValueError(f"Model type {config.model_type} not supported")

def get_attention_layer(config: AutoConfig):
    if config.model_type == "llama":
        return QuantizedLlamaAttention
    elif config.model_type == "qwen3":
        return QuantizedQwen3Attention
    else:
        raise ValueError(f"Model type {config.model_type} not supported")
