# Texture baker

Small texture baker which rasterizes barycentric coordinates to a tensor.
It also implements an interpolation module which can be used to bake attributes to textures then.

## Usage

The baker can quickly bake vertex attributes to the a texture atlas based on the UV coordinates.
It supports baking on the CPU and GPU.

```python
from texture_baker import TextureBaker

mesh = ...
uv = mesh.uv # num_vertex, 2
triangle_idx = mesh.faces # num_faces, 3
vertices = mesh.vertices # num_vertex, 3

tb  = TextureBaker()
# First get the barycentric coordinates
rast = tb.rasterize(
    uv=uv, face_indices=triangle_idx, bake_resolution=1024
)
# Then interpolate vertex attributes
position_bake = tb.interpolate(attr=vertices, rast=rast, face_indices=triangle_idx)
```
