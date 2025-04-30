import torch
from einops import rearrange
from torch import Tensor

from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None) -> Tensor:
    q_shape = q.shape
    k_shape = k.shape

    if pe is not None:
        # Only convert once to v's dtype at the end, not twice
        q = _apply_positional_encoding(q, pe, q_shape).type_as(v)
        k = _apply_positional_encoding(k, pe, k_shape).type_as(v)

    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if comfy.model_management.is_device_mps(pos.device) or comfy.model_management.is_intel_xpu() or comfy.model_management.is_directml_enabled():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.to(dtype=freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def _apply_positional_encoding(t: Tensor, pe: Tensor, target_shape):
    # Combine _to, reshape, and positional encoding computation efficiently
    t = t.reshape(*t.shape[:-1], -1, 2)
    pe_0, pe_1 = pe[..., 0], pe[..., 1]
    t0, t1 = t[..., 0], t[..., 1]
    return (pe_0 * t0 + pe_1 * t1).reshape(target_shape)

