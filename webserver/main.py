import sys

sys.path.append("/webserver")

import rembg
import torch
from PIL import Image
from tqdm import tqdm

from sf3d.system import SF3D
from sf3d.utils import remove_background, resize_foreground

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#on startup
@app.on_event("startup")
def load_model():
    global model
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors")
    model.to(device)
    model.eval()

@app.on_event("shutdown")
def close_model():
    del model

@app.post("/reconstruct_mesh")
async def reconstruct_mesh(file: UploadFile = File(...)):
    rembg_session = rembg.new_session()
    image = remove_background(
        Image.open(file.file).convert("RGBA"), rembg_session
    )
    image = resize_foreground(image, 0.85)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mesh, glob_dict = model.run_image(
                [image],
                bake_resolution=1024,
                remesh="none"
            )
    print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
    mesh.export("./output/mesh.glb", include_normals=True)
    return FileResponse("./output/mesh.glb")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5678)
