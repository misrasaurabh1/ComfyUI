from __future__ import annotations

import os
import base64
import json
import time
import logging
import folder_paths
import glob
import comfy.utils
from aiohttp import web
from PIL import Image
from io import BytesIO
from folder_paths import map_legacy, filter_files_extensions, filter_files_content_types


class ModelFileManager:
    def __init__(self) -> None:
        self.cache: dict[str, tuple[list[dict], dict[str, float], float]] = {}

    def get_cache(self, key: str, default=None) -> tuple[list[dict], dict[str, float], float] | None:
        return self.cache.get(key, default)

    def set_cache(self, key: str, value: tuple[list[dict], dict[str, float], float]):
        self.cache[key] = value

    def clear_cache(self):
        self.cache.clear()

    def add_routes(self, routes):
        # NOTE: This is an experiment to replace `/models`
        @routes.get("/experiment/models")
        async def get_model_folders(request):
            model_types = list(folder_paths.folder_names_and_paths.keys())
            folder_black_list = ["configs", "custom_nodes"]
            output_folders: list[dict] = []
            for folder in model_types:
                if folder in folder_black_list:
                    continue
                output_folders.append({"name": folder, "folders": folder_paths.get_folder_paths(folder)})
            return web.json_response(output_folders)

        # NOTE: This is an experiment to replace `/models/{folder}`
        @routes.get("/experiment/models/{folder}")
        async def get_all_models(request):
            folder = request.match_info.get("folder", None)
            if not folder in folder_paths.folder_names_and_paths:
                return web.Response(status=404)
            files = self.get_model_file_list(folder)
            return web.json_response(files)

        @routes.get("/experiment/models/preview/{folder}/{path_index}/{filename:.*}")
        async def get_model_preview(request):
            folder_name = request.match_info.get("folder", None)
            path_index = int(request.match_info.get("path_index", None))
            filename = request.match_info.get("filename", None)

            if not folder_name in folder_paths.folder_names_and_paths:
                return web.Response(status=404)

            folders = folder_paths.folder_names_and_paths[folder_name]
            folder = folders[0][path_index]
            full_filename = os.path.join(folder, filename)

            previews = self.get_model_previews(full_filename)
            default_preview = previews[0] if len(previews) > 0 else None
            if default_preview is None or (isinstance(default_preview, str) and not os.path.isfile(default_preview)):
                return web.Response(status=404)

            try:
                with Image.open(default_preview) as img:
                    img_bytes = BytesIO()
                    img.save(img_bytes, format="WEBP")
                    img_bytes.seek(0)
                    return web.Response(body=img_bytes.getvalue(), content_type="image/webp")
            except:
                return web.Response(status=404)

    def get_model_file_list(self, folder_name: str):
        folder_name = map_legacy(folder_name)
        folders = folder_paths.folder_names_and_paths[folder_name]
        output_list: list[dict] = []

        for index, folder in enumerate(folders[0]):
            if not os.path.isdir(folder):
                continue

            out = self.cache_model_file_list_(folder)
            if out is None:
                out = self.recursive_search_models_(folder, index)
                self.set_cache(folder, out)
            output_list.extend(out[0])

        return output_list

    def cache_model_file_list_(self, folder: str):
        model_file_list_cache = self.get_cache(folder)
        if model_file_list_cache is None:
            return None
        # Check the folder's mtime directly only once
        try:
            folder_mtime = os.path.getmtime(folder)
        except FileNotFoundError:
            return None
        # Check if folder mtime matches
        if folder_mtime != model_file_list_cache[1].get(folder, None):
            return None
        # Check that all subdirectory mtimes haven't changed
        for subfolder, cached_mtime in model_file_list_cache[1].items():
            try:
                if os.path.getmtime(subfolder) != cached_mtime:
                    return None
            except FileNotFoundError:
                return None
        return model_file_list_cache

    def recursive_search_models_(self, directory: str, pathIndex: int) -> tuple[list[str], dict[str, float], float]:
        if not os.path.isdir(directory):
            return [], {}, time.perf_counter()

        excluded_dir_names = {".git"}
        # TODO use settings
        include_hidden_files = False

        result: list[str] = []
        dirs: dict[str, float] = {}

        # Use os.walk with updated subdirs/filenames filtering
        for dirpath, subdirs, filenames in os.walk(directory, followlinks=True, topdown=True):
            # Filter subdirectories to avoid excluded and hidden dirs
            subdirs[:] = [d for d in subdirs if d not in excluded_dir_names and (include_hidden_files or not d.startswith('.'))]
            # Only include non-hidden files if required
            if not include_hidden_files:
                filenames = [f for f in filenames if not f.startswith('.')]
            # Only include supported file extensions
            filenames = filter_files_extensions(filenames, folder_paths.supported_pt_extensions)
            if filenames:
                dirpath_rel = os.path.relpath(dirpath, directory)
                # Optimize relpath usage for same directory
                if dirpath_rel == '.':
                    rel_result = filenames
                else:
                    rel_result = [os.path.join(dirpath_rel, name) for name in filenames]
                result.extend(rel_result)

            # Add directory mtimes for changed folder detection
            for d in subdirs:
                path = os.path.join(dirpath, d)
                try:
                    dirs[path] = os.path.getmtime(path)
                except FileNotFoundError:
                    logging.warning(f"Warning: Unable to access {path}. Skipping this path.")

        return ([{"name": f, "pathIndex": pathIndex} for f in result], dirs, time.perf_counter())

    def get_model_previews(self, filepath: str) -> list[str | BytesIO]:
        dirname = os.path.dirname(filepath)

        if not os.path.exists(dirname):
            return []

        basename = os.path.splitext(filepath)[0]
        match_files = glob.glob(f"{basename}.*", recursive=False)
        image_files = filter_files_content_types(match_files, "image")
        safetensors_file = next(filter(lambda x: x.endswith(".safetensors"), match_files), None)
        safetensors_metadata = {}

        result: list[str | BytesIO] = []

        for filename in image_files:
            _basename = os.path.splitext(filename)[0]
            if _basename == basename:
                result.append(filename)
            if _basename == f"{basename}.preview":
                result.append(filename)

        if safetensors_file:
            safetensors_filepath = os.path.join(dirname, safetensors_file)
            header = comfy.utils.safetensors_header(safetensors_filepath, max_size=8*1024*1024)
            if header:
                safetensors_metadata = json.loads(header)
        safetensors_images = safetensors_metadata.get("__metadata__", {}).get("ssmd_cover_images", None)
        if safetensors_images:
            safetensors_images = json.loads(safetensors_images)
            for image in safetensors_images:
                result.append(BytesIO(base64.b64decode(image)))

        return result

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear_cache()
