import torch
import comfy.rmsnorm


def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    # If tracing/scripting, force reflect mode for compatibility
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"

    img_shape = img.shape
    ndim = img.ndim
    pad_list = []
    need_padding = False

    # Compute padding for each spatial dimension (assumed trailing spatial dims)
    for i in range(ndim - 2):
        dim_size = img_shape[i + 2]
        psize = patch_size[i]
        pad_amt = (psize - dim_size % psize) % psize
        pad_list.append((0, pad_amt))
        if pad_amt != 0:
            need_padding = True

    if not need_padding:
        return img  # no padding required

    # Flatten pad pairs in reverse order for torch.nn.functional.pad (last dim first)
    pad_args = []
    for low, high in reversed(pad_list):
        pad_args.extend([low, high])

    return torch.nn.functional.pad(img, pad_args, mode=padding_mode)


rms_norm = comfy.rmsnorm.rms_norm
