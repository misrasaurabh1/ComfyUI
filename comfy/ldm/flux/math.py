import torch
from einops import rearrange
from torch import Tensor

from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None) -> Tensor:
    q_shape = q.shape
    k_shape = k.shape

    if pe is not None:
        q = q.to(dtype=pe.dtype).reshape(*q.shape[:-1], -1, 1, 2)
        k = k.to(dtype=pe.dtype).reshape(*k.shape[:-1], -1, 1, 2)
        q = (
            (pe[..., 0] * q[..., 0] + pe[..., 1] * q[..., 1])
            .reshape(*q_shape)
            .type_as(v)
        )
        k = (
            (pe[..., 0] * k[..., 0] + pe[..., 1] * k[..., 1])
            .reshape(*k_shape)
            .type_as(v)
        )

    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    use_cpu_device = (
        comfy.model_management.is_device_mps(pos.device)
        or comfy.model_management.is_intel_xpu()
        or comfy.model_management.is_directml_enabled()
    )
    device = torch.device("cpu") if use_cpu_device else pos.device

    scale = torch.linspace(
        0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float32, device=device
    )  # Use float32 to save memory and utilize better GPU performance
    omega = 1.0 / (theta**scale)
    out = torch.einsum(
        "...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega
    )
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    out_stack = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out_rearranged = rearrange(out_stack, "b n d (i j) -> b n d i j", i=2, j=2)

    # Directly convert to float32 and back to the original device
    return out_rearranged.to(dtype=torch.float32, device=pos.device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.to(dtype=freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
