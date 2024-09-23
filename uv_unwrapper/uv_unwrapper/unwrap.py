import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Unwrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def _box_assign_vertex_to_cube_face(
        self,
        vertex_positions: Tensor,
        vertex_normals: Tensor,
        triangle_idxs: Tensor,
        bbox: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Assigns each vertex to a cube face based on the face normal

        Args:
            vertex_positions (Tensor, Nv 3, float): Vertex positions
            vertex_normals (Tensor, Nv 3, float): Vertex normals
            triangle_idxs (Tensor, Nf 3, int): Triangle indices
            bbox (Tensor, 2 3, float): Bounding box of the mesh

        Returns:
            Tensor, Nf 3 2, float: UV coordinates
            Tensor, Nf, int: Cube face indices
        """

        # Test to not have a scaled model to fit the space better
        # bbox_min = bbox[:1].mean(-1, keepdim=True)
        # bbox_max = bbox[1:].mean(-1, keepdim=True)
        # v_pos_normalized = (vertex_positions - bbox_min) / (bbox_max - bbox_min)

        # Create a [0, 1] normalized vertex position
        v_pos_normalized = (vertex_positions - bbox[:1]) / (bbox[1:] - bbox[:1])
        # And to [-1, 1]
        v_pos_normalized = 2.0 * v_pos_normalized - 1.0

        # Get all vertex positions for each triangle
        # Now how do we define to which face the triangle belongs? Mean face pos? Max vertex pos?
        v0 = v_pos_normalized[triangle_idxs[:, 0]]
        v1 = v_pos_normalized[triangle_idxs[:, 1]]
        v2 = v_pos_normalized[triangle_idxs[:, 2]]
        tri_stack = torch.stack([v0, v1, v2], dim=1)

        vn0 = vertex_normals[triangle_idxs[:, 0]]
        vn1 = vertex_normals[triangle_idxs[:, 1]]
        vn2 = vertex_normals[triangle_idxs[:, 2]]
        tri_stack_nrm = torch.stack([vn0, vn1, vn2], dim=1)

        # Just average the normals per face
        face_normal = F.normalize(torch.sum(tri_stack_nrm, 1), eps=1e-6, dim=-1)

        # Now decide based on the face normal in which box map we project
        # abs_x, abs_y, abs_z = tri_stack_nrm.abs().unbind(-1)
        abs_x, abs_y, abs_z = tri_stack.abs().unbind(-1)

        axis = torch.tensor(
            [
                [1, 0, 0],  # 0
                [-1, 0, 0],  # 1
                [0, 1, 0],  # 2
                [0, -1, 0],  # 3
                [0, 0, 1],  # 4
                [0, 0, -1],  # 5
            ],
            device=face_normal.device,
            dtype=face_normal.dtype,
        )
        face_normal_axis = (face_normal[:, None] * axis[None]).sum(-1)
        index = face_normal_axis.argmax(-1)

        max_axis, uc, vc = (
            torch.ones_like(abs_x),
            torch.zeros_like(tri_stack[..., :1]),
            torch.zeros_like(tri_stack[..., :1]),
        )
        mask_pos_x = index == 0
        max_axis[mask_pos_x] = abs_x[mask_pos_x]
        uc[mask_pos_x] = tri_stack[mask_pos_x][..., 1:2]
        vc[mask_pos_x] = -tri_stack[mask_pos_x][..., -1:]

        mask_neg_x = index == 1
        max_axis[mask_neg_x] = abs_x[mask_neg_x]
        uc[mask_neg_x] = tri_stack[mask_neg_x][..., 1:2]
        vc[mask_neg_x] = -tri_stack[mask_neg_x][..., -1:]

        mask_pos_y = index == 2
        max_axis[mask_pos_y] = abs_y[mask_pos_y]
        uc[mask_pos_y] = tri_stack[mask_pos_y][..., 0:1]
        vc[mask_pos_y] = -tri_stack[mask_pos_y][..., -1:]

        mask_neg_y = index == 3
        max_axis[mask_neg_y] = abs_y[mask_neg_y]
        uc[mask_neg_y] = tri_stack[mask_neg_y][..., 0:1]
        vc[mask_neg_y] = -tri_stack[mask_neg_y][..., -1:]

        mask_pos_z = index == 4
        max_axis[mask_pos_z] = abs_z[mask_pos_z]
        uc[mask_pos_z] = tri_stack[mask_pos_z][..., 0:1]
        vc[mask_pos_z] = tri_stack[mask_pos_z][..., 1:2]

        mask_neg_z = index == 5
        max_axis[mask_neg_z] = abs_z[mask_neg_z]
        uc[mask_neg_z] = tri_stack[mask_neg_z][..., 0:1]
        vc[mask_neg_z] = -tri_stack[mask_neg_z][..., 1:2]

        # UC from [-1, 1] to [0, 1]
        max_dim_div = max_axis.max(dim=0, keepdim=True).values
        uc = ((uc[..., 0] / max_dim_div + 1.0) * 0.5).clip(0, 1)
        vc = ((vc[..., 0] / max_dim_div + 1.0) * 0.5).clip(0, 1)

        uv = torch.stack([uc, vc], dim=-1)

        return uv, index

    def _assign_faces_uv_to_atlas_index(
        self,
        vertex_positions: Tensor,
        triangle_idxs: Tensor,
        face_uv: Tensor,
        face_index: Tensor,
    ) -> Tensor:  # noqa: F821
        """
        Assigns the face UV to the atlas index

        Args:
            vertex_positions (Float[Tensor, "Nv 3"]): Vertex positions
            triangle_idxs (Integer[Tensor, "Nf 3"]): Triangle indices
            face_uv (Float[Tensor, "Nf 3 2"]): Face UV coordinates
            face_index (Integer[Tensor, "Nf"]): Face indices

        Returns:
            Integer[Tensor, "Nf"]: Atlas index
        """
        return torch.ops.UVUnwrapper.assign_faces_uv_to_atlas_index(
            vertex_positions.cpu(),
            triangle_idxs.cpu(),
            face_uv.view(-1, 2).cpu(),
            face_index.cpu(),
        ).to(vertex_positions.device)

    def _find_slice_offset_and_scale(
        self, index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:  # noqa: F821
        """
        Find the slice offset and scale

        Args:
            index (Integer[Tensor, "Nf"]): Atlas index

        Returns:
            Float[Tensor, "Nf"]: Offset x
            Float[Tensor, "Nf"]: Offset y
            Float[Tensor, "Nf"]: Division x
            Float[Tensor, "Nf"]: Division y
        """

        # 6 due to the 6 cube faces
        off = 1 / 3
        dupl_off = 1 / 6

        # Here, we need to decide how to pack the textures in the case of overlap
        def x_offset_calc(x, i):
            offset_calc = i // 6
            # Initial coordinates - just 3x2 grid
            if offset_calc == 0:
                return off * x
            else:
                # Smaller 3x2 grid plus eventual shift to right for
                # second overlap
                return dupl_off * x + min(offset_calc - 1, 1) * 0.5

        def y_offset_calc(x, i):
            offset_calc = i // 6
            # Initial coordinates - just a 3x2 grid
            if offset_calc == 0:
                return off * x
            else:
                # Smaller coordinates in the lowest row
                return dupl_off * x + off * 2

        offset_x = torch.zeros_like(index, dtype=torch.float32)
        offset_y = torch.zeros_like(index, dtype=torch.float32)
        offset_x_vals = [0, 1, 2, 0, 1, 2]
        offset_y_vals = [0, 0, 0, 1, 1, 1]
        for i in range(index.max().item() + 1):
            mask = index == i
            if not mask.any():
                continue
            offset_x[mask] = x_offset_calc(offset_x_vals[i % 6], i)
            offset_y[mask] = y_offset_calc(offset_y_vals[i % 6], i)

        div_x = torch.full_like(index, 6 // 2, dtype=torch.float32)
        # All overlap elements are saved in half scale
        div_x[index >= 6] = 6
        div_y = div_x.clone()  # Same for y
        # Except for the random overlaps
        div_x[index >= 12] = 2
        # But the random overlaps are saved in a large block in the lower thirds
        div_y[index >= 12] = 3

        return offset_x, offset_y, div_x, div_y

    def _calculate_tangents(
        self,
        vertex_positions: Tensor,
        vertex_normals: Tensor,
        triangle_idxs: Tensor,
        face_uv: Tensor,
    ) -> Tensor:
        """
        Calculate the tangents for each triangle

        Args:
            vertex_positions (Float[Tensor, "Nv 3"]): Vertex positions
            vertex_normals (Float[Tensor, "Nv 3"]): Vertex normals
            triangle_idxs (Integer[Tensor, "Nf 3"]): Triangle indices
            face_uv (Float[Tensor, "Nf 3 2"]): Face UV coordinates

        Returns:
            Float[Tensor, "Nf 3 4"]: Tangents
        """
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = face_uv.unbind(1)
        for i in range(0, 3):
            pos[i] = vertex_positions[triangle_idxs[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = triangle_idxs[:, i]

        if(torch.backends.mps.is_available()):
            tangents = torch.zeros_like(vertex_normals).contiguous()
            tansum = torch.zeros_like(vertex_normals).contiguous()
        else:
            tangents = torch.zeros_like(vertex_normals)
            tansum = torch.zeros_like(vertex_normals)

        # Compute tangent space for each triangle
        duv1 = tex[1] - tex[0]
        duv2 = tex[2] - tex[0]
        dpos1 = pos[1] - pos[0]
        dpos2 = pos[2] - pos[0]

        tng_nom = dpos1 * duv2[..., 1:2] - dpos2 * duv1[..., 1:2]

        denom = duv1[..., 0:1] * duv2[..., 1:2] - duv1[..., 1:2] * duv2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        denom_safe = denom.clip(1e-6)
        tang = tng_nom / denom_safe

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        # Also normalize it. Here we do not normalize the individual triangles first so larger area
        # triangles influence the tangent space more
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(
            tangents
            - (tangents * vertex_normals).sum(-1, keepdim=True) * vertex_normals
        )

        return tangents

    def _rotate_uv_slices_consistent_space(
        self,
        vertex_positions: Tensor,
        vertex_normals: Tensor,
        triangle_idxs: Tensor,
        uv: Tensor,
        index: Tensor,
    ) -> Tensor:
        """
        Rotate the UV slices so they are in a consistent space

        Args:
            vertex_positions (Float[Tensor, "Nv 3"]): Vertex positions
            vertex_normals (Float[Tensor, "Nv 3"]): Vertex normals
            triangle_idxs (Integer[Tensor, "Nf 3"]): Triangle indices
            uv (Float[Tensor, "Nf 3 2"]): UV coordinates
            index (Integer[Tensor, "Nf"]): Atlas index

        Returns:
            Float[Tensor, "Nf 3 2"]: Rotated UV coordinates
        """

        tangents = self._calculate_tangents(
            vertex_positions, vertex_normals, triangle_idxs, uv
        )
        pos_stack = torch.stack(
            [
                -vertex_positions[..., 1],
                vertex_positions[..., 0],
                torch.zeros_like(vertex_positions[..., 0]),
            ],
            dim=-1,
        )
        expected_tangents = F.normalize(
            torch.linalg.cross(
                vertex_normals,
                torch.linalg.cross(pos_stack, vertex_normals, dim=-1),
                dim=-1,
            ),
            -1,
        )

        actual_tangents = tangents[triangle_idxs]
        expected_tangents = expected_tangents[triangle_idxs]

        def rotation_matrix_2d(theta):
            c, s = torch.cos(theta), torch.sin(theta)
            return torch.tensor([[c, -s], [s, c]])

        # Now find the rotation
        index_mod = index % 6  # Shouldn't happen. Just for safety
        for i in range(6):
            mask = index_mod == i
            if not mask.any():
                continue

            actual_mean_tangent = actual_tangents[mask].mean(dim=(0, 1))
            expected_mean_tangent = expected_tangents[mask].mean(dim=(0, 1))

            dot_product = torch.dot(actual_mean_tangent, expected_mean_tangent)
            cross_product = (
                actual_mean_tangent[0] * expected_mean_tangent[1]
                - actual_mean_tangent[1] * expected_mean_tangent[0]
            )
            angle = torch.atan2(cross_product, dot_product)

            rot_matrix = rotation_matrix_2d(angle).to(mask.device)
            # Center the uv coordinate to be in the range of -1 to 1 and 0 centered
            uv_cur = uv[mask] * 2 - 1  # Center it first
            # Rotate it
            uv[mask] = torch.einsum("ij,nfj->nfi", rot_matrix, uv_cur)

            # Rescale uv[mask] to be within the 0-1 range
            uv[mask] = (uv[mask] - uv[mask].min()) / (uv[mask].max() - uv[mask].min())

        return uv

    def _handle_slice_uvs(
        self,
        uv: Tensor,
        index: Tensor,  # noqa: F821
        island_padding: float,
        max_index: int = 6 * 2,
    ) -> Tensor:  # noqa: F821
        """
        Handle the slice UVs

        Args:
            uv (Float[Tensor, "Nf 3 2"]): UV coordinates
            index (Integer[Tensor, "Nf"]): Atlas index
            island_padding (float): Island padding
            max_index (int): Maximum index

        Returns:
            Float[Tensor, "Nf 3 2"]: Updated UV coordinates

        """
        uc, vc = uv.unbind(-1)

        # Get the second slice (The first overlap)
        index_filter = [index == i for i in range(6, max_index)]

        # Normalize them to always fully fill the atlas patch
        for i, fi in enumerate(index_filter):
            if fi.sum() > 0:
                # Scale the slice but only up to a factor of 2
                # This keeps the texture resolution with the first slice in line (Half space in UV)
                uc[fi] = (uc[fi] - uc[fi].min()) / (uc[fi].max() - uc[fi].min()).clip(
                    0.5
                )
                vc[fi] = (vc[fi] - vc[fi].min()) / (vc[fi].max() - vc[fi].min()).clip(
                    0.5
                )

        uc_padded = (uc * (1 - 2 * island_padding) + island_padding).clip(0, 1)
        vc_padded = (vc * (1 - 2 * island_padding) + island_padding).clip(0, 1)

        return torch.stack([uc_padded, vc_padded], dim=-1)

    def _handle_remaining_uvs(
        self,
        uv: Tensor,
        index: Tensor,  # noqa: F821
        island_padding: float,
    ) -> Tensor:
        """
        Handle the remaining UVs (The ones that are not slices)

        Args:
            uv (Float[Tensor, "Nf 3 2"]): UV coordinates
            index (Integer[Tensor, "Nf"]): Atlas index
            island_padding (float): Island padding

        Returns:
            Float[Tensor, "Nf 3 2"]: Updated UV coordinates
        """
        uc, vc = uv.unbind(-1)
        # Get all remaining elements
        remaining_filter = index >= 6 * 2
        squares_left = remaining_filter.sum()

        if squares_left == 0:
            return uv

        uc = uc[remaining_filter]
        vc = vc[remaining_filter]

        # Or remaining triangles are distributed in a rectangle
        # The rectangle takes 0.5 of the entire uv space in width and 1/3 in height
        ratio = 0.5 * (1 / 3)  # 1.5
        # sqrt(744/(0.5*(1/3)))

        mult = math.sqrt(squares_left / ratio)
        num_square_width = int(math.ceil(0.5 * mult))
        num_square_height = int(math.ceil(squares_left / num_square_width))

        width = 1 / num_square_width
        height = 1 / num_square_height

        # The idea is again to keep the texture resolution consistent with the first slice
        # This only occupys half the region in the texture chart but the scaling on the squares
        # assumes full coverage.
        clip_val = min(width, height) * 1.5
        # Now normalize the UVs with taking into account the maximum scaling
        uc = (uc - uc.min(dim=1, keepdim=True).values) / (
            uc.amax(dim=1, keepdim=True) - uc.amin(dim=1, keepdim=True)
        ).clip(clip_val)
        vc = (vc - vc.min(dim=1, keepdim=True).values) / (
            vc.amax(dim=1, keepdim=True) - vc.amin(dim=1, keepdim=True)
        ).clip(clip_val)
        # Add a small padding
        uc = (
            uc * (1 - island_padding * num_square_width * 0.5)
            + island_padding * num_square_width * 0.25
        ).clip(0, 1)
        vc = (
            vc * (1 - island_padding * num_square_height * 0.5)
            + island_padding * num_square_height * 0.25
        ).clip(0, 1)

        uc = uc * width
        vc = vc * height

        # And calculate offsets for each element
        idx = torch.arange(uc.shape[0], device=uc.device, dtype=torch.int32)
        x_idx = idx % num_square_width
        y_idx = idx // num_square_width
        # And move each triangle to its own spot
        uc = uc + x_idx[:, None] * width
        vc = vc + y_idx[:, None] * height

        uc = (uc * (1 - 2 * island_padding * 0.5) + island_padding * 0.5).clip(0, 1)
        vc = (vc * (1 - 2 * island_padding * 0.5) + island_padding * 0.5).clip(0, 1)

        uv[remaining_filter] = torch.stack([uc, vc], dim=-1)

        return uv

    def _distribute_individual_uvs_in_atlas(
        self,
        face_uv: Tensor,
        assigned_faces: Tensor,
        offset_x: Tensor,
        offset_y: Tensor,
        div_x: Tensor,
        div_y: Tensor,
        island_padding: float,
    ) -> Tensor:
        """
        Distribute the individual UVs in the atlas

        Args:
            face_uv (Float[Tensor, "Nf 3 2"]): Face UV coordinates
            assigned_faces (Integer[Tensor, "Nf"]): Assigned faces
            offset_x (Float[Tensor, "Nf"]): Offset x
            offset_y (Float[Tensor, "Nf"]): Offset y
            div_x (Float[Tensor, "Nf"]): Division x
            div_y (Float[Tensor, "Nf"]): Division y
            island_padding (float): Island padding

        Returns:
            Float[Tensor, "Nf 3 2"]: Updated UV coordinates
        """
        # Place the slice first
        placed_uv = self._handle_slice_uvs(face_uv, assigned_faces, island_padding)
        # Then handle the remaining overlap elements
        placed_uv = self._handle_remaining_uvs(
            placed_uv, assigned_faces, island_padding
        )

        uc, vc = placed_uv.unbind(-1)
        uc = uc / div_x[:, None] + offset_x[:, None]
        vc = vc / div_y[:, None] + offset_y[:, None]

        uv = torch.stack([uc, vc], dim=-1).view(-1, 2)

        return uv

    def _get_unique_face_uv(
        self,
        uv: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get the unique face UV

        Args:
            uv (Float[Tensor, "Nf 3 2"]): UV coordinates

        Returns:
            Float[Tensor, "Utex 3"]: Unique UV coordinates
            Integer[Tensor, "Nf"]: Vertex index
        """
        unique_uv, unique_idx = torch.unique(uv, return_inverse=True, dim=0)
        # And add the face to uv index mapping
        vtex_idx = unique_idx.view(-1, 3)

        return unique_uv, vtex_idx

    def _align_mesh_with_main_axis(
        self, vertex_positions: Tensor, vertex_normals: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Align the mesh with the main axis

        Args:
            vertex_positions (Float[Tensor, "Nv 3"]): Vertex positions
            vertex_normals (Float[Tensor, "Nv 3"]): Vertex normals

        Returns:
            Float[Tensor, "Nv 3"]: Rotated vertex positions
            Float[Tensor, "Nv 3"]: Rotated vertex normals
        """

        # Use pca to find the 2 main axis (third is derived by cross product)
        # Set the random seed so it's repeatable
        torch.manual_seed(0)
        _, _, v = torch.pca_lowrank(vertex_positions, q=2)
        main_axis, seconday_axis = v[:, 0], v[:, 1]

        main_axis = F.normalize(main_axis, eps=1e-6, dim=-1)  # 3,
        # Orthogonalize the second axis
        seconday_axis = F.normalize(
            seconday_axis
            - (seconday_axis * main_axis).sum(-1, keepdim=True) * main_axis,
            eps=1e-6,
            dim=-1,
        )  # 3,
        # Create perpendicular third axis
        third_axis = F.normalize(
            torch.cross(main_axis, seconday_axis, dim=-1), dim=-1, eps=1e-6
        )  # 3,

        # Check to which canonical axis each aligns
        main_axis_max_idx = main_axis.abs().argmax().item()
        seconday_axis_max_idx = seconday_axis.abs().argmax().item()
        third_axis_max_idx = third_axis.abs().argmax().item()

        # Now sort the axes based on the argmax so they align with thecanonoical axes
        # If two axes have the same argmax move one of them
        all_possible_axis = {0, 1, 2}
        cur_index = 1
        while (
            len(set([main_axis_max_idx, seconday_axis_max_idx, third_axis_max_idx]))
            != 3
        ):
            # Find missing axis
            missing_axis = all_possible_axis - set(
                [main_axis_max_idx, seconday_axis_max_idx, third_axis_max_idx]
            )
            missing_axis = missing_axis.pop()
            # Just assign it to third axis as it had the smallest contribution to the
            # overall shape
            if cur_index == 1:
                third_axis_max_idx = missing_axis
            elif cur_index == 2:
                seconday_axis_max_idx = missing_axis
            else:
                raise ValueError("Could not find 3 unique axis")
            cur_index += 1

        if len({main_axis_max_idx, seconday_axis_max_idx, third_axis_max_idx}) != 3:
            raise ValueError("Could not find 3 unique axis")

        axes = [None] * 3
        axes[main_axis_max_idx] = main_axis
        axes[seconday_axis_max_idx] = seconday_axis
        axes[third_axis_max_idx] = third_axis
        # Create rotation matrix from the individual axes
        rot_mat = torch.stack(axes, dim=1).T

        # Now rotate the vertex positions and vertex normals so the mesh aligns with the main axis
        vertex_positions = torch.einsum("ij,nj->ni", rot_mat, vertex_positions)
        vertex_normals = torch.einsum("ij,nj->ni", rot_mat, vertex_normals)

        return vertex_positions, vertex_normals

    def forward(
        self,
        vertex_positions: Tensor,
        vertex_normals: Tensor,
        triangle_idxs: Tensor,
        island_padding: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Unwrap the mesh

        Args:
            vertex_positions (Float[Tensor, "Nv 3"]): Vertex positions
            vertex_normals (Float[Tensor, "Nv 3"]): Vertex normals
            triangle_idxs (Integer[Tensor, "Nf 3"]): Triangle indices
            island_padding (float): Island padding

        Returns:
            Float[Tensor, "Utex 3"]: Unique UV coordinates
            Integer[Tensor, "Nf"]: Vertex index
        """
        vertex_positions, vertex_normals = self._align_mesh_with_main_axis(
            vertex_positions, vertex_normals
        )
        bbox = torch.stack(
            [vertex_positions.min(dim=0).values, vertex_positions.max(dim=0).values],
            dim=0,
        )  # 2, 3

        face_uv, face_index = self._box_assign_vertex_to_cube_face(
            vertex_positions, vertex_normals, triangle_idxs, bbox
        )

        face_uv = self._rotate_uv_slices_consistent_space(
            vertex_positions, vertex_normals, triangle_idxs, face_uv, face_index
        )

        assigned_atlas_index = self._assign_faces_uv_to_atlas_index(
            vertex_positions, triangle_idxs, face_uv, face_index
        )

        offset_x, offset_y, div_x, div_y = self._find_slice_offset_and_scale(
            assigned_atlas_index
        )

        placed_uv = self._distribute_individual_uvs_in_atlas(
            face_uv,
            assigned_atlas_index,
            offset_x,
            offset_y,
            div_x,
            div_y,
            island_padding,
        )

        return self._get_unique_face_uv(placed_uv)
