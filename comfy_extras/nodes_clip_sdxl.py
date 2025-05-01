from nodes import MAX_RESOLUTION

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
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens, add_dict={"aesthetic_score": ascore, "width": width, "height": height}), )

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
        tokens_l = clip.tokenize(text_l)["l"]

        len_g = len(tokens["g"])
        len_l = len(tokens_l)

        # If lengths match, just use directly for speed
        if len_l == len_g:
            tokens["l"] = tokens_l
        else:
            # Only tokenize("") once per call
            empty = clip.tokenize("")
            empty_l = empty["l"]
            empty_g = empty["g"]

            # Avoid slow "+=" growing: slice and multiply
            if len_l < len_g:
                # Pad tokens_l efficiently
                padding = empty_l * ((len_g - len_l + len(empty_l) - 1) // len(empty_l))
                tokens["l"] = tokens_l + padding[:len_g - len_l]
            else: # len_l > len_g
                padding = empty_g * ((len_l - len_g + len(empty_g) - 1) // len(empty_g))
                tokens["g"] += padding[:len_l - len_g]
                tokens["l"] = tokens_l

        return (clip.encode_from_tokens_scheduled(
            tokens,
            add_dict={
                "width": width,
                "height": height,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "target_width": target_width,
                "target_height": target_height
            }
        ), )

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeSDXLRefiner": CLIPTextEncodeSDXLRefiner,
    "CLIPTextEncodeSDXL": CLIPTextEncodeSDXL,
}
