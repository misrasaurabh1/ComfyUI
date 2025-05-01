from nodes import MAX_RESOLUTION
from functools import lru_cache

class CLIPTextEncodeSDXLRefiner:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, ascore, width, height, text):
        # Use cached tokenizer per clip instance to avoid recomputing
        tokenize = self._get_text_tokenizer(clip)
        tokens = tokenize(text)
        # Pass tokens forward as before
        return (
            clip.encode_from_tokens_scheduled(
                tokens,
                add_dict={"aesthetic_score": ascore, "width": width, "height": height}
            ),
        )

    def _get_text_tokenizer(self, clip):
        # Get or create a memoized tokenize function per-clip instance.
        if not hasattr(clip, "_tokenize_cache"):
            @lru_cache(maxsize=32)
            def tokenize_cached(text):
                return clip.tokenize(text)
            clip._tokenize_cache = tokenize_cached
        return clip._tokenize_cache

class CLIPTextEncodeSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text_g": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "text_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        return (clip.encode_from_tokens_scheduled(tokens, add_dict={"width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}), )

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeSDXLRefiner": CLIPTextEncodeSDXLRefiner,
    "CLIPTextEncodeSDXL": CLIPTextEncodeSDXL,
}
