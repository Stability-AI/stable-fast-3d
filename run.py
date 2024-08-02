import argparse
import os

import rembg
import torch
from PIL import Image
from tqdm import tqdm

from sf3d.system import SF3D
from sf3d.utils import remove_background, resize_foreground

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", type=str, nargs="+", help="Path to input image(s) or folder."
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to use. If no CUDA-compatible device is found, the baking will fail. Default: 'cuda:0'",
    )
    parser.add_argument(
        "--pretrained-model",
        default="stabilityai/stable-fast-3d",
        type=str,
        help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/stable-fast-3d'",
    )
    parser.add_argument(
        "--foreground-ratio",
        default=0.85,
        type=float,
        help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
    )
    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        help="Output directory to save the results. Default: 'output/'",
    )
    parser.add_argument(
        "--texture-resolution",
        default=1024,
        type=int,
        help="Texture atlas resolution. Default: 1024",
    )
    parser.add_argument(
        "--remesh_option",
        choices=["none", "triangle", "quad"],
        default="none",
        help="Remeshing option",
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for inference"
    )
    args = parser.parse_args()

    # Ensure args.device contains cuda
    if "cuda" not in args.device:
        raise ValueError(
            "CUDA device is required for baking and hence running the method."
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"

    model = SF3D.from_pretrained(
        args.pretrained_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(device)
    model.eval()

    rembg_session = rembg.new_session()
    images = []
    idx = 0
    for image_path in args.image:

        def handle_image(image_path, idx):
            image = remove_background(
                Image.open(image_path).convert("RGBA"), rembg_session
            )
            image = resize_foreground(image, args.foreground_ratio)
            os.makedirs(os.path.join(output_dir, str(idx)), exist_ok=True)
            image.save(os.path.join(output_dir, str(idx), "input.png"))
            images.append(image)

        if os.path.isdir(image_path):
            image_paths = [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
            for image_path in image_paths:
                handle_image(image_path, idx)
                idx += 1
        else:
            handle_image(image_path, idx)
            idx += 1

    for i in tqdm(range(0, len(images), args.batch_size)):
        image = images[i : i + args.batch_size]
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                mesh, glob_dict = model.run_image(
                    image,
                    bake_resolution=args.texture_resolution,
                    remesh=args.remesh_option,
                )
        print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

        if len(image) == 1:
            out_mesh_path = os.path.join(output_dir, str(i), "mesh.glb")
            mesh.export(out_mesh_path, include_normals=True)
        else:
            for j in range(len(mesh)):
                out_mesh_path = os.path.join(output_dir, str(i + j), "mesh.glb")
                mesh[j].export(out_mesh_path, include_normals=True)
