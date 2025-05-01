import torch
from PIL import Image
from comfy.cli_args import args, LatentPreviewMethod
from comfy.taesd.taesd import TAESD
import comfy.model_management
import folder_paths
import comfy.utils
import logging

MAX_PREVIEW_RESOLUTION = args.preview_size

def preview_to_image(latent_image):
        # change scale from -1..1 to 0..1, then to 0..255
        latents_ubyte = ((latent_image + 1.0) / 2.0).clamp_(0, 1).mul_(0xFF)
        # Only .to(dtype=torch.uint8) if not already uint8
        if comfy.model_management.directml_enabled:
                if latents_ubyte.dtype != torch.uint8:
                        latents_ubyte = latents_ubyte.to(dtype=torch.uint8)
        # Only .to(device="cpu", dtype=torch.uint8) if not already
        nb = comfy.model_management.device_supports_non_blocking(latent_image.device)
        if not (latents_ubyte.device.type == 'cpu' and latents_ubyte.dtype == torch.uint8):
                latents_ubyte = latents_ubyte.to(device="cpu", dtype=torch.uint8, non_blocking=nb)
        # Expect contiguous for numpy
        if not latents_ubyte.is_contiguous():
                latents_ubyte = latents_ubyte.contiguous()
        return Image.fromarray(latents_ubyte.numpy())

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)

class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1])[0].movedim(0, 2)
        return preview_to_image(x_sample)


class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None):
            t_factors = torch.tensor(latent_rgb_factors, device="cpu")
            self.latent_rgb_factors_cpu = t_factors.transpose(0, 1).contiguous()
            self.latent_rgb_factors = self.latent_rgb_factors_cpu  # always keep a CPU ref for fast move
            self.latent_rgb_factors_bias_cpu = None
            self.latent_rgb_factors_bias = None
            if latent_rgb_factors_bias is not None:
                    t_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")
                    self.latent_rgb_factors_bias_cpu = t_bias
                    self.latent_rgb_factors_bias = t_bias

    def decode_latent_to_preview(self, x0):
            # Move factors & bias to x0's dtype/device, only if needed
            if (self.latent_rgb_factors.device != x0.device or 
                self.latent_rgb_factors.dtype != x0.dtype):
                    self.latent_rgb_factors = self.latent_rgb_factors_cpu.to(dtype=x0.dtype, device=x0.device)
            if self.latent_rgb_factors_bias_cpu is not None:
                    if (self.latent_rgb_factors_bias is None or 
                        self.latent_rgb_factors_bias.device != x0.device or 
                        self.latent_rgb_factors_bias.dtype != x0.dtype):
                            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias_cpu.to(dtype=x0.dtype, device=x0.device)

            # Fast axis selection
            if x0.ndim == 5:
                    x0_mat = x0[0, :, 0]
            else:
                    x0_mat = x0[0]
            # Linear RGB mapping, fast path
            latent_image = torch.nn.functional.linear(
                x0_mat.movedim(0, -1), 
                self.latent_rgb_factors, 
                bias=self.latent_rgb_factors_bias
            )
            return preview_to_image(latent_image)


def get_previewer(device, latent_format):
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer methods
        taesd_decoder_path = None
        if latent_format.taesd_decoder_name is not None:
            taesd_decoder_path = next(
                (fn for fn in folder_paths.get_filename_list("vae_approx")
                    if fn.startswith(latent_format.taesd_decoder_name)),
                ""
            )
            taesd_decoder_path = folder_paths.get_full_path("vae_approx", taesd_decoder_path)

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB

        if method == LatentPreviewMethod.TAESD:
            if taesd_decoder_path:
                taesd = TAESD(None, taesd_decoder_path, latent_channels=latent_format.latent_channels).to(device)
                previewer = TAESDPreviewerImpl(taesd)
            else:
                logging.warning("Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(latent_format.taesd_decoder_name))

        if previewer is None:
            if latent_format.latent_rgb_factors is not None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors, latent_format.latent_rgb_factors_bias)
    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer(model.load_device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

