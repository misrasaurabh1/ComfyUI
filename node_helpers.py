import hashlib
import torch

from comfy.cli_args import args

from PIL import ImageFile, UnidentifiedImageError

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c

def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError): #PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x

def hasher():
    hashfuncs = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    return hashfuncs[args.default_hashing_function]

def string_to_torch_dtype(string):
    if string == "fp32":
        return torch.float32
    if string == "fp16":
        return torch.float16
    if string == "bf16":
        return torch.bfloat16

def image_alpha_fix(destination, source):
    if destination.shape[-1] < source.shape[-1]:
        source = source[..., :destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        # Only pad if necessary
        padded = torch.nn.functional.pad(destination, (0, 1))
        padded[..., -1] = 1.0
        destination = padded
    return destination, source

def _fast_crop_centered(samples, old_width, old_height, width, height):
    old_aspect = old_width / old_height
    new_aspect = width / height
    x = 0
    y = 0
    if old_aspect > new_aspect:
        cropw = round(old_width - old_width * (new_aspect / old_aspect))
        x = cropw // 2
        w = old_width - cropw
        h = old_height
    elif old_aspect < new_aspect:
        croph = round(old_height - old_height * (old_aspect / new_aspect))
        y = croph // 2
        h = old_height - croph
        w = old_width
    else:
        h = old_height
        w = old_width
    # Use narrow for efficiency and avoid tensor copy where possible
    return samples.narrow(-2, y, h).narrow(-1, x, w)
