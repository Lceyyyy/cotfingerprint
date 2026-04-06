from types import MethodType
from typing import TYPE_CHECKING

import torch
import torch.nn as nn


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments


def _resolve_num_heads(attn_module: nn.Module) -> int | None:
    for attr in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
        value = getattr(attn_module, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    config = getattr(attn_module, "config", None)
    for attr in ("num_attention_heads", "num_heads", "n_heads", "n_head"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _resolve_head_dim(attn_module: nn.Module, hidden_size: int, num_heads: int) -> int | None:
    head_dim = getattr(attn_module, "head_dim", None)
    if isinstance(head_dim, int) and head_dim > 0:
        return head_dim
    if hidden_size % num_heads != 0:
        return None
    return hidden_size // num_heads


def _resolve_hidden_size(attn_module: nn.Module) -> int | None:
    value = getattr(attn_module, "hidden_size", None)
    if isinstance(value, int) and value > 0:
        return value

    config = getattr(attn_module, "config", None)
    value = getattr(config, "hidden_size", None)
    if isinstance(value, int) and value > 0:
        return value

    q_proj = getattr(attn_module, "q_proj", None)
    if isinstance(q_proj, nn.Linear):
        if isinstance(q_proj.in_features, int) and q_proj.in_features > 0:
            return q_proj.in_features
        if isinstance(q_proj.out_features, int) and q_proj.out_features > 0:
            return q_proj.out_features

    return None


def _is_patchable_attention(module: nn.Module) -> bool:
    required = ("q_proj", "k_proj", "v_proj", "o_proj")
    return all(hasattr(module, name) for name in required)


def _init_head_gate_mlp(mlp: nn.Sequential, gate_bias_init: float, gate_init_noise_std: float) -> None:
    for sub_module in mlp:
        if isinstance(sub_module, nn.Linear):
            nn.init.zeros_(sub_module.weight)
            nn.init.zeros_(sub_module.bias)
            if gate_init_noise_std > 0:
                nn.init.normal_(sub_module.weight, mean=0.0, std=gate_init_noise_std)
    last_linear = mlp[-1]
    if isinstance(last_linear, nn.Linear):
        last_linear.bias.data.fill_(gate_bias_init)


def apply_head_gate_patch(model: nn.Module, finetuning_args: "FinetuningArguments") -> int:
    patched_count = 0
    for module in model.modules():
        if not _is_patchable_attention(module):
            continue
        if getattr(module, "_head_gate_patched", False):
            continue

        num_heads = _resolve_num_heads(module)
        if num_heads is None:
            continue

        hidden_size = _resolve_hidden_size(module)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            continue

        head_dim = _resolve_head_dim(module, hidden_size, num_heads)
        if head_dim is None:
            continue

        gate_hidden_dim = finetuning_args.gate_hidden_dim or head_dim
        gate_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_dim, gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(gate_hidden_dim, 1),
                )
                for _ in range(num_heads)
            ]
        )
        q_proj = getattr(module, "q_proj", None)
        if isinstance(q_proj, nn.Linear):
            gate_mlps = gate_mlps.to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)
        for mlp in gate_mlps:
            _init_head_gate_mlp(mlp, finetuning_args.gate_bias_init, finetuning_args.gate_init_noise_std)

        module.head_gate_mlps = gate_mlps
        module.last_head_gate = None
        module.injected_head_gate = None
        module._head_gate_num_heads = num_heads
        module._head_gate_head_dim = head_dim

        original_forward = module.forward

        def wrapped_forward(self, hidden_states, *args, original_forward=original_forward, **kwargs):
            outputs = original_forward(hidden_states, *args, **kwargs)
            if isinstance(outputs, torch.Tensor):
                attn_output = outputs
                extra_outputs = None
            elif isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                attn_output = outputs[0]
                extra_outputs = outputs[1:]
            else:
                return outputs

            if not (isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3 and attn_output.ndim == 3):
                return outputs

            bsz, seq_len, hidden = attn_output.shape
            expected_hidden = self._head_gate_num_heads * self._head_gate_head_dim
            if hidden != expected_hidden:
                return outputs

            x_heads = hidden_states.reshape(bsz, seq_len, self._head_gate_num_heads, self._head_gate_head_dim)
            per_head_gates = []
            for head_idx in range(self._head_gate_num_heads):
                head_feature = x_heads[:, :, head_idx, :]
                gate = torch.sigmoid(self.head_gate_mlps[head_idx](head_feature))  # [B, T, 1]
                per_head_gates.append(gate.unsqueeze(1))  # [B, 1, T, 1]
            head_gate = torch.cat(per_head_gates, dim=1)  # [B, H, T, 1]

            gate_to_apply = head_gate
            if self.injected_head_gate is not None:
                injected_gate = self.injected_head_gate.to(device=head_gate.device, dtype=head_gate.dtype)
                if injected_gate.shape != head_gate.shape:
                    raise ValueError(
                        f"injected_head_gate shape mismatch: expected={tuple(head_gate.shape)}, got={tuple(injected_gate.shape)}"
                    )
                gate_to_apply = injected_gate
            self.last_head_gate = gate_to_apply

            attn_heads = attn_output.reshape(bsz, seq_len, self._head_gate_num_heads, self._head_gate_head_dim).transpose(1, 2)
            gated_output = (attn_heads * gate_to_apply).transpose(1, 2).reshape(bsz, seq_len, hidden)

            if extra_outputs is None:
                return gated_output
            return (gated_output, *extra_outputs)

        module.forward = MethodType(wrapped_forward, module)
        module._head_gate_patched = True
        patched_count += 1

    return patched_count


def setup_gate_trainable_params(model: nn.Module) -> int:
    for param in model.parameters():
        param.requires_grad_(False)

    trainable = 0
    for module in model.modules():
        gate_mlps = getattr(module, "head_gate_mlps", None)
        if gate_mlps is None:
            continue
        for param in gate_mlps.parameters():
            param.requires_grad_(True)
            trainable += param.numel()
    return trainable


def compute_gate_mean(model: nn.Module) -> torch.Tensor | None:
    means: list[torch.Tensor] = []
    for module in model.modules():
        gate = getattr(module, "last_head_gate", None)
        if isinstance(gate, torch.Tensor):
            means.append(gate.mean())

    if not means:
        return None
    return torch.stack(means).mean()
