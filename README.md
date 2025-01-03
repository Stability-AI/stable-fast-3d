# SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement

<a href="https://arxiv.org/abs/2408.00653"><img src="https://img.shields.io/badge/Arxiv-2408.00653-B31B1B.svg"></a> <a href="https://huggingface.co/stabilityai/stable-fast-3d"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a> <a href="https://huggingface.co/spaces/stabilityai/stable-fast-3d"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>

<div align="center">
  <img src="demo_files/teaser.gif" alt="Teaser">
</div>

<br>

This is the official codebase for **Stable Fast 3D**, a state-of-the-art open-source model for **fast** feedforward 3D mesh reconstruction from a single image.

<br>

<p align="center">
    <img width="450" src="demo_files/comp.gif"/>
</p>

<p align="center">
    <img width="450" src="demo_files/scatterplot.jpg"/>
</p>

Stable Fast 3D is based on [TripoSR](https://github.com/VAST-AI-Research/TripoSR) but introduces several new key techniques. For one, we explicitly optimize our model to produce good meshes without artifacts alongside textures with UV unwrapping. We also delight the color and predict material parameters so the assets can be easily integrated into a game. We achieve all of this while still maintaining the fast inference speeds of TripoSR.

## Getting Started

### Installation

Ensure your environment is:
- Python >= 3.8
- Optional: CUDA or MPS has to be available
- For Windows **(experimental)**: Visual Studio 2022
- Has PyTorch installed according to your platform: https://pytorch.org/get-started/locally/ [Make sure the Pytorch CUDA version matches your system's.]
- Update setuptools by `pip install -U setuptools==69.5.1`
- Install wheel by `pip install wheel`

Then, install the remaining requirements with `pip install -r requirements.txt`.
For the gradio demo, an additional `pip install -r requirements-demo.txt` is required.

### Requesting Access and Login

Our model is gated at [Hugging Face](https://huggingface.co):

1. Log in to Hugging Face and request access [here](https://huggingface.co/stabilityai/stable-fast-3d).
2. Create an access token with read permissions [here](https://huggingface.co/settings/tokens).
3. Run `huggingface-cli login` in the environment and enter the token.

### Support for MPS (for Mac Silicon) **(experimental)**

Stable Fast 3D can also run on Macs via the MPS backend, with the texture baker using custom metal kernels similar to the corresponding CUDA kernels.

Note that support is **experimental** and not guaranteed to give the same performance and/or quality as the CUDA backend.

You will need to install OpenMP runtime to enable clang support for `-fopenmp`. Follow the tutorial here https://mac.r-project.org/openmp/ 

MPS backend support was tested on M1 max 64GB with the latest PyTorch nightly release. We recommend you install the latest PyTorch (2.4.0 as of writing) and/or the nightly version to avoid any issues that my arise with older PyTorch versions.

You also need to run the code with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

MPS currently consumes more memory compared to the CUDA PyTorch backend. We recommend running the CPU version if your system has less than 32GB of unified memory.

### Windows Support **(experimental)**

To run Stable Fast 3D on Windows, you must install Visual Studio (currently tested on VS 2022) and the appropriate PyTorch and CUDA versions.
Then, follow the installation steps as mentioned above.

Note that Windows support is **experimental** and not guaranteed to give the same performance and/or quality as Linux.

### CPU Support

CPU backend will automatically be used if no GPU is detected in your system.

If you have a GPU but are facing issues and want to use the CPU backend instead, set the environment variable `SF3D_USE_CPU=1` to force the CPU backend.

### Manual Inference

```sh
python run.py demo_files/examples/chair1.png --output-dir output/
```
This will save the reconstructed 3D model as a GLB file to `output/`. You can also specify more than one image path separated by spaces. The default options takes about **6GB VRAM** for a single image input.

You may also use `--texture-resolution` to specify the resolution in pixels of the output texture and `--remesh_option` to specify the remeshing operation (None, Triangle, Quad).

For detailed usage of this script, use `python run.py --help`.

### Local Gradio App

```sh
python gradio_app.py
```


## ComfyUI extension

Custom nodes and an [example workflow](./demo_files/workflows/sf3d_example.json) are provided for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

To install:

* Clone this repo into ```custom_nodes```:
 ```shell
  $ cd ComfyUI/custom_nodes
  $ git clone https://github.com/Stability-AI/stable-fast-3d
 ```
* Install dependencies:
 ```shell
  $ cd stable-fast-3d
  $ pip install -r requirements.txt
 ```
* Restart ComfyUI

## Remesher Options:

  -`none`: mesh unchanged after generation. No CPU overhead.

  -`triangle`: verticies and edges are rearranged to form a triangle topography. Implementation is from: *"[A Remeshing Approach to Multiresolution Modeling](https://github.com/sgsellan/botsch-kobbelt-remesher-libigl)" by M. Botsch and L. Kobbelt*. CPU overhead expected.

  -`quad`: verticies and edges are rearanged in quadrilateral topography with a proper quad flow. The quad mesh is split into triangles for export with GLB. Implementation is from *"[Instant Field-Aligned Meshes](https://github.com/wjakob/instant-meshes)" from Jakob et al.*. CPU overhead expected.

Additionally the target vertex count can be specified. This is not a hard constraint but a rough vertex count the method aims to create. This target is ignored if the remesher is set to `none`.

## Citation
```BibTeX
@article{sf3d2024,
  title={SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement},
  author={Boss, Mark and Huang, Zixuan and Vasishta, Aaryaman and Jampani, Varun},
  journal={arXiv preprint},
  year={2024}
}
```
