import base64
import os
import io
from io import BytesIO
from contextlib import nullcontext
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from PIL import Image
import rembg
import torch
from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground

load_dotenv()

app = FastAPI()

model = SF3D.from_pretrained(
    os.getenv("PRETRAINED_MODEL"),
    config_name="config.yaml",
    weight_name="model.safetensors",
)
model.to(os.getenv("DEVICE"))
model.eval()

rembg_session = rembg.new_session()

@app.post("/generate",
    responses = { 200: { "content": { "model/gltf-binary": {} } } },
    response_class=Response
)
async def generate(file: UploadFile):
    # load the image using Pillow
    image = Image.open(file.file)
    image = remove_bg(image)
    mesh = generate_model(image)

    # return the image as a binary stream with a suitable content-disposition header for download
    return Response(
        content=mesh,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": "attachment; filename=mesh.glb"},
    )

def remove_bg(img: Image) -> Image:
    img = remove_background(img, rembg_session)
    img = resize_foreground(img, float(os.getenv("FOREGROUND_RATIO")))
    return img

def generate_model(image: Image):
    device = os.getenv("DEVICE")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        with torch.autocast(
            device_type=device, dtype=torch.float16
        ) if "cuda" in device else nullcontext():
            mesh, glob_dict = model.run_image(
                image,
                bake_resolution=int(os.getenv("TEXTURE_RESOLUTION")),
                remesh=os.getenv("REMESSH_OPTION"),
                vertex_count=int(os.getenv("TARGET_VERTEX_COUNT")),
            )
    if torch.cuda.is_available():
        print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
    elif torch.backends.mps.is_available():
        print(
            "Peak Memory:", torch.mps.driver_allocated_memory() / 1024 / 1024, "MB"
        )

    return mesh.export(include_normals=True, file_type='glb')
