import base64
import logging
import os
import sys

import comfy.model_management
import folder_paths
import numpy as np
import torch
import trimesh
from PIL import Image
from trimesh.exchange import gltf

sys.path.append(os.path.dirname(__file__))
from sf3d.system import SF3D
from sf3d.utils import resize_foreground

SF3D_CATEGORY = "StableFast3D"
SF3D_MODEL_NAME = "stabilityai/stable-fast-3d"


class StableFast3DLoader:
    CATEGORY = SF3D_CATEGORY
    FUNCTION = "load"
    RETURN_NAMES = ("sf3d_model",)
    RETURN_TYPES = ("SF3D_MODEL",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load(self):
        device = comfy.model_management.get_torch_device()
        model = SF3D.from_pretrained(
            SF3D_MODEL_NAME,
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        model.to(device)
        model.eval()

        return (model,)


class StableFast3DPreview:
    CATEGORY = SF3D_CATEGORY
    FUNCTION = "preview"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mesh": ("MESH",)}}

    def preview(self, mesh):
        glbs = []
        for m in mesh:
            scene = trimesh.Scene(m)
            glb_data = gltf.export_glb(scene, include_normals=True)
            glb_base64 = base64.b64encode(glb_data).decode("utf-8")
            glbs.append(glb_base64)
        return {"ui": {"glbs": glbs}}


class StableFast3DSampler:
    CATEGORY = SF3D_CATEGORY
    FUNCTION = "predict"
    RETURN_NAMES = ("mesh",)
    RETURN_TYPES = ("MESH",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("SF3D_MODEL",),
                "image": ("IMAGE",),
                "foreground_ratio": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "texture_resolution": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 2048, "step": 256},
                ),
            },
            "optional": {
                "mask": ("MASK",),
                "remesh": (["none", "triangle", "quad"],),
                "vertex_count": (
                    "INT",
                    {"default": -1, "min": -1, "max": 20000, "step": 1},
                ),
            },
        }

    def predict(
        s,
        model,
        image,
        mask,
        foreground_ratio,
        texture_resolution,
        remesh="none",
        vertex_count=-1,
    ):
        if image.shape[0] != 1:
            raise ValueError("Only one image can be processed at a time")

        pil_image = Image.fromarray(
            torch.clamp(torch.round(255.0 * image[0]), 0, 255)
            .type(torch.uint8)
            .cpu()
            .numpy()
        )

        if mask is not None:
            print("Using Mask")
            mask_np = np.clip(255.0 * mask[0].detach().cpu().numpy(), 0, 255).astype(
                np.uint8
            )
            mask_pil = Image.fromarray(mask_np, mode="L")
            pil_image.putalpha(mask_pil)
        else:
            if image.shape[3] != 4:
                print("No mask or alpha channel detected, Converting to RGBA")
                pil_image = pil_image.convert("RGBA")

        pil_image = resize_foreground(pil_image, foreground_ratio)
        print(remesh)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                mesh, glob_dict = model.run_image(
                    pil_image,
                    bake_resolution=texture_resolution,
                    remesh=remesh,
                    vertex_count=vertex_count,
                )

        if mesh.vertices.shape[0] == 0:
            raise ValueError("No subject detected in the image")

        return ([mesh],)


class StableFast3DSave:
    CATEGORY = SF3D_CATEGORY
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "filename_prefix": ("STRING", {"default": "SF3D"}),
            }
        }

    def __init__(self):
        self.type = "output"

    def save(self, mesh, filename_prefix):
        output_dir = folder_paths.get_output_directory()
        glbs = []
        for idx, m in enumerate(mesh):
            scene = trimesh.Scene(m)
            glb_data = gltf.export_glb(scene, include_normals=True)
            logging.info(f"Generated GLB model with {len(glb_data)} bytes")

            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(filename_prefix, output_dir)
            )
            filename = filename.replace("%batch_num%", str(idx))
            out_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.glb")
            with open(out_path, "wb") as f:
                f.write(glb_data)
            glbs.append(base64.b64encode(glb_data).decode("utf-8"))
        return {"ui": {"glbs": glbs}}


NODE_DISPLAY_NAME_MAPPINGS = {
    "StableFast3DLoader": "Stable Fast 3D Loader",
    "StableFast3DPreview": "Stable Fast 3D Preview",
    "StableFast3DSampler": "Stable Fast 3D Sampler",
    "StableFast3DSave": "Stable Fast 3D Save",
}

NODE_CLASS_MAPPINGS = {
    "StableFast3DLoader": StableFast3DLoader,
    "StableFast3DPreview": StableFast3DPreview,
    "StableFast3DSampler": StableFast3DSampler,
    "StableFast3DSave": StableFast3DSave,
}

WEB_DIRECTORY = "./comfyui"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
