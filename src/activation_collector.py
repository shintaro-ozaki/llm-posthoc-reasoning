import torch
from abc import ABC, abstractmethod


class BaseActivationCollector(ABC):
    """
    Base class for activation collectors.
    - layer-indexed storage (None-safe)
    - dtype-safe (fp16 for numpy)
    """

    def __init__(self, model, dtype=torch.float16):
        self.model = model
        self.layers = model.model.layers
        self.num_layers = len(self.layers)
        self.save_dtype = torch.float16
        self.attn_output = [None] * self.num_layers
        self.mlp_output = [None] * self.num_layers
        self.proj_output = {}

        self.handles = []

    def _to_cpu(self, x):
        if x is None:
            return None

        # gpt-oss / MoE 対応
        if isinstance(x, tuple):
            x = x[0]  # hidden_states のみ使う

        return x.detach().to("cpu").to(self.save_dtype)

    def _store(self, key, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            return
        if key not in self.proj_output:
            self.proj_output[key] = []
        self.proj_output[key].append(self._to_cpu(out))

    def _register_attn_o_proj(self, layer_idx, layer):
        def hook(_, __, out):
            self.attn_output[layer_idx] = self._to_cpu(out)

        self.handles.append(layer.self_attn.o_proj.register_forward_hook(hook))

    def _register_mlp_block(self, layer_idx, layer):
        def hook(_, __, out):
            self.mlp_output[layer_idx] = self._to_cpu(out)

        self.handles.append(layer.mlp.register_forward_hook(hook))

    @abstractmethod
    def register(self):
        pass

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class DenseMLPCollector(BaseActivationCollector):
    """
    gate / up / down を持つ dense MLP 用
    """

    def register(self):
        for i, layer in enumerate(self.layers):
            self._register_attn_o_proj(i, layer)
            self._register_mlp_block(i, layer)

            for name in ["gate_proj", "up_proj", "down_proj"]:
                mod = getattr(layer.mlp, name)
                key = f"layer{i}/mlp.{name}"
                self.handles.append(
                    mod.register_forward_hook(
                        lambda m, inp, out, k=key: self._store(k, out)
                    )
                )


class Qwen2ActivationCollector(DenseMLPCollector):
    pass


class LlamaActivationCollector(DenseMLPCollector):
    pass


class Qwen3ActivationCollector(DenseMLPCollector):
    pass


class Phi3ActivationCollector(BaseActivationCollector):
    def register(self):
        for i, layer in enumerate(self.layers):
            self._register_attn_o_proj(i, layer)
            self._register_mlp_block(i, layer)

            for name in ["gate_up_proj", "down_proj"]:
                mod = getattr(layer.mlp, name)
                key = f"layer{i}/mlp.{name}"
                self.handles.append(
                    mod.register_forward_hook(
                        lambda m, inp, out, k=key: self._store(k, out)
                    )
                )


class GptOssActivationCollector(BaseActivationCollector):
    def register(self):
        for i, layer in enumerate(self.layers):
            self._register_attn_o_proj(i, layer)
            self._register_mlp_block(i, layer)

            if hasattr(layer.mlp, "router"):
                key = f"layer{i}/mlp.router"
                self.handles.append(
                    layer.mlp.router.register_forward_hook(
                        lambda m, inp, out, k=key: self._store(k, out)
                    )
                )
