from PIL import Image
import numpy as np
import comfy.utils
import time

#You can use this node to save full size images through the websocket, the
#images will be sent in exactly the same format as the image previews: as
#binary images on the websocket with a 8 byte header indicating the type
#of binary message (first 4 bytes) and the image format (next 4 bytes).

#Note that no metadata will be put in the images saved with this node.

class SaveImageWebsocket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),}
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "api/image"

    def save_images(self, images):
        n_images = images.shape[0]
        pbar = comfy.utils.ProgressBar(n_images)
        
        # Transfer all to CPU and numpy in one operation
        np_images = images.cpu().numpy() # (N, ...)

        # Scale and clip all in a vectorized manner
        i = np_images * 255.
        i = np.clip(i, 0, 255).astype(np.uint8)
        
        for idx, img_arr in enumerate(i):
            img = Image.fromarray(img_arr)
            # Only pass total once; ProgressBar ignores if no change.
            pbar.update_absolute(idx, n_images if idx == 0 else None, ("PNG", img, None))

        return {}

    @classmethod
    def IS_CHANGED(s, images):
        return time.time()

NODE_CLASS_MAPPINGS = {
    "SaveImageWebsocket": SaveImageWebsocket,
}
