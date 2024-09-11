#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <torch/extension.h>
#ifndef __ARM_ARCH_ISA_A64
#include <immintrin.h>
#endif

#include "baker.h"

// #define TIMING
#define BINS 8

namespace texture_baker_cpp {
// Calculate the centroid of a triangle
tb_float2 triangle_centroid(const tb_float2 &v0, const tb_float2 &v1,
                            const tb_float2 &v2) {
  return {(v0.x + v1.x + v2.x) * 0.3333f, (v0.y + v1.y + v2.y) * 0.3333f};
}

float BVH::find_best_split_plane(const BVHNode &node, int &best_axis,
                                 int &best_pos, AABB &centroidBounds) {
  float best_cost = std::numeric_limits<float>::max();

  for (int axis = 0; axis < 2; ++axis) // We use 2 as we have only x and y
  {
    float boundsMin = centroidBounds.min[axis];
    float boundsMax = centroidBounds.max[axis];
    if (boundsMin == boundsMax) {
      continue;
    }

    // Populate the bins
    float scale = BINS / (boundsMax - boundsMin);
    float leftCountArea[BINS - 1], rightCountArea[BINS - 1];
    int leftSum = 0, rightSum = 0;

#ifndef __ARM_ARCH_ISA_A64
#ifndef _MSC_VER
    if (__builtin_cpu_supports("sse"))
#elif (defined(_M_AMD64) || defined(_M_X64))
    // SSE supported on Windows
    if constexpr (true)
#endif
    {
      __m128 min4[BINS], max4[BINS];
      unsigned int count[BINS];
      for (unsigned int i = 0; i < BINS; i++)
        min4[i] = _mm_set_ps1(1e30f), max4[i] = _mm_set_ps1(-1e30f),
        count[i] = 0;
      for (int i = node.start; i < node.end; i++) {
        int tri_idx = triangle_indices[i];
        const Triangle &triangle = triangles[tri_idx];

        int binIdx = std::min(
            BINS - 1, (int)((triangle.centroid[axis] - boundsMin) * scale));
        count[binIdx]++;
        __m128 v0 = _mm_set_ps(triangle.v0.x, triangle.v0.y, 0.0f, 0.0f);
        __m128 v1 = _mm_set_ps(triangle.v1.x, triangle.v1.y, 0.0f, 0.0f);
        __m128 v2 = _mm_set_ps(triangle.v2.x, triangle.v2.y, 0.0f, 0.0f);
        min4[binIdx] = _mm_min_ps(min4[binIdx], v0);
        max4[binIdx] = _mm_max_ps(max4[binIdx], v0);
        min4[binIdx] = _mm_min_ps(min4[binIdx], v1);
        max4[binIdx] = _mm_max_ps(max4[binIdx], v1);
        min4[binIdx] = _mm_min_ps(min4[binIdx], v2);
        max4[binIdx] = _mm_max_ps(max4[binIdx], v2);
      }
      // gather data for the 7 planes between the 8 bins
      __m128 leftMin4 = _mm_set_ps1(1e30f), rightMin4 = leftMin4;
      __m128 leftMax4 = _mm_set_ps1(-1e30f), rightMax4 = leftMax4;
      for (int i = 0; i < BINS - 1; i++) {
        leftSum += count[i];
        rightSum += count[BINS - 1 - i];
        leftMin4 = _mm_min_ps(leftMin4, min4[i]);
        rightMin4 = _mm_min_ps(rightMin4, min4[BINS - 2 - i]);
        leftMax4 = _mm_max_ps(leftMax4, max4[i]);
        rightMax4 = _mm_max_ps(rightMax4, max4[BINS - 2 - i]);
        float le[4], re[4];
        _mm_store_ps(le, _mm_sub_ps(leftMax4, leftMin4));
        _mm_store_ps(re, _mm_sub_ps(rightMax4, rightMin4));
        // SSE order goes from back to front
        leftCountArea[i] = leftSum * (le[2] * le[3]); // 2D area calculation
        rightCountArea[BINS - 2 - i] =
            rightSum * (re[2] * re[3]); // 2D area calculation
      }
    }
#else
    if constexpr (false) {
    }
#endif
    else {
      struct Bin {
        AABB bounds;
        int triCount = 0;
      } bins[BINS];

      for (int i = node.start; i < node.end; i++) {
        int tri_idx = triangle_indices[i];
        const Triangle &triangle = triangles[tri_idx];

        int binIdx = std::min(
            BINS - 1, (int)((triangle.centroid[axis] - boundsMin) * scale));
        bins[binIdx].triCount++;
        bins[binIdx].bounds.grow(triangle.v0);
        bins[binIdx].bounds.grow(triangle.v1);
        bins[binIdx].bounds.grow(triangle.v2);
      }

      // Gather data for the planes between the bins
      AABB leftBox, rightBox;

      for (int i = 0; i < BINS - 1; i++) {
        leftSum += bins[i].triCount;
        leftBox.grow(bins[i].bounds);
        leftCountArea[i] = leftSum * leftBox.area();

        rightSum += bins[BINS - 1 - i].triCount;
        rightBox.grow(bins[BINS - 1 - i].bounds);
        rightCountArea[BINS - 2 - i] = rightSum * rightBox.area();
      }
    }

    // Calculate SAH cost for the planes
    scale = (boundsMax - boundsMin) / BINS;
    for (int i = 0; i < BINS - 1; i++) {
      float planeCost = leftCountArea[i] + rightCountArea[i];
      if (planeCost < best_cost) {
        best_axis = axis;
        best_pos = i + 1;
        best_cost = planeCost;
      }
    }
  }

  return best_cost;
}

void BVH::update_node_bounds(BVHNode &node, AABB &centroidBounds) {
#ifndef __ARM_ARCH_ISA_A64
#ifndef _MSC_VER
  if (__builtin_cpu_supports("sse"))
#elif (defined(_M_AMD64) || defined(_M_X64))
  // SSE supported on Windows
  if constexpr (true)
#endif
  {
    __m128 min4 = _mm_set_ps1(1e30f), max4 = _mm_set_ps1(-1e30f);
    __m128 cmin4 = _mm_set_ps1(1e30f), cmax4 = _mm_set_ps1(-1e30f);

    for (int i = node.start; i < node.end; i += 2) {
      int tri_idx1 = triangle_indices[i];
      const Triangle &leafTri1 = triangles[tri_idx1];
      // Check if the second actually exists in the node
      __m128 v0, v1, v2, centroid;
      if (i + 1 < node.end) {
        int tri_idx2 = triangle_indices[i + 1];
        const Triangle leafTri2 = triangles[tri_idx2];

        v0 = _mm_set_ps(leafTri1.v0.x, leafTri1.v0.y, leafTri2.v0.x,
                        leafTri2.v0.y);
        v1 = _mm_set_ps(leafTri1.v1.x, leafTri1.v1.y, leafTri2.v1.x,
                        leafTri2.v1.y);
        v2 = _mm_set_ps(leafTri1.v2.x, leafTri1.v2.y, leafTri2.v2.x,
                        leafTri2.v2.y);
        centroid = _mm_set_ps(leafTri1.centroid.x, leafTri1.centroid.y,
                              leafTri2.centroid.x, leafTri2.centroid.y);
      } else {
        // Otherwise do some duplicated work
        v0 = _mm_set_ps(leafTri1.v0.x, leafTri1.v0.y, leafTri1.v0.x,
                        leafTri1.v0.y);
        v1 = _mm_set_ps(leafTri1.v1.x, leafTri1.v1.y, leafTri1.v1.x,
                        leafTri1.v1.y);
        v2 = _mm_set_ps(leafTri1.v2.x, leafTri1.v2.y, leafTri1.v2.x,
                        leafTri1.v2.y);
        centroid = _mm_set_ps(leafTri1.centroid.x, leafTri1.centroid.y,
                              leafTri1.centroid.x, leafTri1.centroid.y);
      }

      min4 = _mm_min_ps(min4, v0);
      max4 = _mm_max_ps(max4, v0);
      min4 = _mm_min_ps(min4, v1);
      max4 = _mm_max_ps(max4, v1);
      min4 = _mm_min_ps(min4, v2);
      max4 = _mm_max_ps(max4, v2);
      cmin4 = _mm_min_ps(cmin4, centroid);
      cmax4 = _mm_max_ps(cmax4, centroid);
    }

    float min_values[4], max_values[4], cmin_values[4], cmax_values[4];
    _mm_store_ps(min_values, min4);
    _mm_store_ps(max_values, max4);
    _mm_store_ps(cmin_values, cmin4);
    _mm_store_ps(cmax_values, cmax4);

    node.bbox.min.x = std::min(min_values[3], min_values[1]);
    node.bbox.min.y = std::min(min_values[2], min_values[0]);
    node.bbox.max.x = std::max(max_values[3], max_values[1]);
    node.bbox.max.y = std::max(max_values[2], max_values[0]);

    centroidBounds.min.x = std::min(cmin_values[3], cmin_values[1]);
    centroidBounds.min.y = std::min(cmin_values[2], cmin_values[0]);
    centroidBounds.max.x = std::max(cmax_values[3], cmax_values[1]);
    centroidBounds.max.y = std::max(cmax_values[2], cmax_values[0]);
  }
#else
  if constexpr (false) {
  }
#endif
  {
    node.bbox.invalidate();
    centroidBounds.invalidate();

    // Calculate the bounding box for the node
    for (int i = node.start; i < node.end; ++i) {
      int tri_idx = triangle_indices[i];
      const Triangle &tri = triangles[tri_idx];
      node.bbox.grow(tri.v0);
      node.bbox.grow(tri.v1);
      node.bbox.grow(tri.v2);
      centroidBounds.grow(tri.centroid);
    }
  }
}

void BVH::build(const tb_float2 *vertices, const tb_int3 *indices,
                const int64_t &num_indices) {
#ifdef TIMING
  auto start = std::chrono::high_resolution_clock::now();
#endif
  // Create triangles
  for (size_t i = 0; i < num_indices; ++i) {
    tb_int3 idx = indices[i];
    triangles.push_back(
        {vertices[idx.x], vertices[idx.y], vertices[idx.z], static_cast<int>(i),
         triangle_centroid(vertices[idx.x], vertices[idx.y], vertices[idx.z])});
  }

  // Initialize triangle_indices
  triangle_indices.resize(triangles.size());
  std::iota(triangle_indices.begin(), triangle_indices.end(), 0);

  // Build BVH nodes
  // Reserve extra capacity to fix windows specific crashes
  nodes.reserve(triangles.size() * 2 + 1);
  nodes.push_back({}); // Create the root node
  root = 0;

  // Define a struct for queue entries
  struct QueueEntry {
    int node_idx;
    int start;
    int end;
  };

  // Queue for breadth-first traversal
  std::queue<QueueEntry> node_queue;
  node_queue.push({root, 0, (int)triangles.size()});

  // Process each node in the queue
  while (!node_queue.empty()) {
    QueueEntry current = node_queue.front();
    node_queue.pop();

    int node_idx = current.node_idx;
    int start = current.start;
    int end = current.end;

    BVHNode &node = nodes[node_idx];
    node.start = start;
    node.end = end;

    // Calculate the bounding box for the node
    AABB centroidBounds;
    update_node_bounds(node, centroidBounds);

    // Determine the best split using SAH
    int best_axis, best_pos;

    float splitCost =
        find_best_split_plane(node, best_axis, best_pos, centroidBounds);
    float nosplitCost = node.calculate_node_cost();

    // Stop condition: if the best cost is greater than or equal to the parent's
    // cost
    if (splitCost >= nosplitCost) {
      // Leaf node
      node.left = node.right = -1;
      continue;
    }

    float scale =
        BINS / (centroidBounds.max[best_axis] - centroidBounds.min[best_axis]);
    int i = node.start;
    int j = node.end - 1;

    // Sort the triangle_indices in the range [start, end) based on the best
    // axis
    while (i <= j) {
      // use the exact calculation we used for binning to prevent rare
      // inaccuracies
      int tri_idx = triangle_indices[i];
      tb_float2 tcentr = triangles[tri_idx].centroid;
      int binIdx = std::min(
          BINS - 1,
          (int)((tcentr[best_axis] - centroidBounds.min[best_axis]) * scale));
      if (binIdx < best_pos)
        i++;
      else
        std::swap(triangle_indices[i], triangle_indices[j--]);
    }
    int leftCount = i - node.start;
    if (leftCount == 0 || leftCount == node.num_triangles()) {
      // Leaf node
      node.left = node.right = -1;
      continue;
    }

    int mid = i;

    // Create and set left child
    node.left = nodes.size();
    nodes.push_back({});
    node_queue.push({node.left, start, mid});

    // Create and set right child
    node = nodes[node_idx]; // Update the node - Potentially stale reference
    node.right = nodes.size();
    nodes.push_back({});
    node_queue.push({node.right, mid, end});
  }
#ifdef TIMING
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "BVH build time: " << elapsed.count() << "s" << std::endl;
#endif
}

// Utility function to clamp a value between a minimum and a maximum
float clamp(float val, float minVal, float maxVal) {
  return std::min(std::max(val, minVal), maxVal);
}

// Function to check if a point (xy) is inside a triangle defined by vertices
// v1, v2, v3
bool barycentric_coordinates(tb_float2 xy, tb_float2 v1, tb_float2 v2,
                             tb_float2 v3, float &u, float &v, float &w) {
  // Vectors from v1 to v2, v3 and xy
  tb_float2 v1v2 = {v2.x - v1.x, v2.y - v1.y};
  tb_float2 v1v3 = {v3.x - v1.x, v3.y - v1.y};
  tb_float2 xyv1 = {xy.x - v1.x, xy.y - v1.y};

  // Dot products of the vectors
  float d00 = v1v2.x * v1v2.x + v1v2.y * v1v2.y;
  float d01 = v1v2.x * v1v3.x + v1v2.y * v1v3.y;
  float d11 = v1v3.x * v1v3.x + v1v3.y * v1v3.y;
  float d20 = xyv1.x * v1v2.x + xyv1.y * v1v2.y;
  float d21 = xyv1.x * v1v3.x + xyv1.y * v1v3.y;

  // Calculate the barycentric coordinates
  float denom = d00 * d11 - d01 * d01;
  v = (d11 * d20 - d01 * d21) / denom;
  w = (d00 * d21 - d01 * d20) / denom;
  u = 1.0f - v - w;

  // Check if the point is inside the triangle
  return (v >= 0.0f) && (w >= 0.0f) && (v + w <= 1.0f);
}

bool BVH::intersect(const tb_float2 &point, float &u, float &v, float &w,
                    int &index) const {
  const int max_stack_size = 64;
  int node_stack[max_stack_size];
  int stack_size = 0;

  node_stack[stack_size++] = root;

  while (stack_size > 0) {
    int node_idx = node_stack[--stack_size];
    const BVHNode &node = nodes[node_idx];

    if (node.is_leaf()) {
      for (int i = node.start; i < node.end; ++i) {
        const Triangle &tri = triangles[triangle_indices[i]];
        if (barycentric_coordinates(point, tri.v0, tri.v1, tri.v2, u, v, w)) {
          index = tri.index;
          return true;
        }
      }
    } else {
      if (nodes[node.right].bbox.overlaps(point)) {
        if (stack_size < max_stack_size) {
          node_stack[stack_size++] = node.right;
        } else {
          // Handle stack overflow
          throw std::runtime_error("Node stack overflow");
        }
      }
      if (nodes[node.left].bbox.overlaps(point)) {
        if (stack_size < max_stack_size) {
          node_stack[stack_size++] = node.left;
        } else {
          // Handle stack overflow
          throw std::runtime_error("Node stack overflow");
        }
      }
    }
  }

  return false;
}

torch::Tensor rasterize_cpu(torch::Tensor uv, torch::Tensor indices,
                            int64_t bake_resolution) {
  int width = bake_resolution;
  int height = bake_resolution;
  int num_pixels = width * height;
  torch::Tensor rast_result = torch::empty(
      {bake_resolution, bake_resolution, 4},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  float *rast_result_ptr = rast_result.contiguous().data_ptr<float>();
  const tb_float2 *vertices = (tb_float2 *)uv.data_ptr<float>();
  const tb_int3 *tris = (tb_int3 *)indices.data_ptr<int>();

  BVH bvh;
  bvh.build(vertices, tris, indices.size(0));

#ifdef TIMING
  auto start = std::chrono::high_resolution_clock::now();
#endif

#pragma omp parallel for
  for (int idx = 0; idx < num_pixels; ++idx) {
    int x = idx / height;
    int y = idx % height;
    int idx_ = idx * 4; // Note: *4 because we're storing float4 per pixel

    tb_float2 pixel_coord = {float(y) / height, float(x) / width};
    pixel_coord.x = clamp(pixel_coord.x, 0.0f, 1.0f);
    pixel_coord.y = 1.0f - clamp(pixel_coord.y, 0.0f, 1.0f);

    float u, v, w;
    int triangle_idx;
    if (bvh.intersect(pixel_coord, u, v, w, triangle_idx)) {
      rast_result_ptr[idx_ + 0] = u;
      rast_result_ptr[idx_ + 1] = v;
      rast_result_ptr[idx_ + 2] = w;
      rast_result_ptr[idx_ + 3] = static_cast<float>(triangle_idx);
    } else {
      rast_result_ptr[idx_ + 0] = 0.0f;
      rast_result_ptr[idx_ + 1] = 0.0f;
      rast_result_ptr[idx_ + 2] = 0.0f;
      rast_result_ptr[idx_ + 3] = -1.0f;
    }
  }

#ifdef TIMING
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Rasterization time: " << elapsed.count() << "s" << std::endl;
#endif
  return rast_result;
}

torch::Tensor interpolate_cpu(torch::Tensor attr, torch::Tensor indices,
                              torch::Tensor rast) {
#ifdef TIMING
  auto start = std::chrono::high_resolution_clock::now();
#endif
  int height = rast.size(0);
  int width = rast.size(1);
  torch::Tensor pos_bake = torch::empty(
      {height, width, 3},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  const float *attr_ptr = attr.contiguous().data_ptr<float>();
  const int *indices_ptr = indices.contiguous().data_ptr<int>();
  const float *rast_ptr = rast.contiguous().data_ptr<float>();
  float *output_ptr = pos_bake.contiguous().data_ptr<float>();

  int num_pixels = width * height;

#pragma omp parallel for
  for (int idx = 0; idx < num_pixels; ++idx) {
    int idx_ = idx * 4; // Index into the float4 array (4 floats per pixel)
    tb_float3 barycentric = {
        rast_ptr[idx_ + 0],
        rast_ptr[idx_ + 1],
        rast_ptr[idx_ + 2],
    };
    int triangle_idx = static_cast<int>(rast_ptr[idx_ + 3]);

    if (triangle_idx < 0) {
      output_ptr[idx * 3 + 0] = 0.0f;
      output_ptr[idx * 3 + 1] = 0.0f;
      output_ptr[idx * 3 + 2] = 0.0f;
      continue;
    }

    tb_int3 triangle = {indices_ptr[3 * triangle_idx + 0],
                        indices_ptr[3 * triangle_idx + 1],
                        indices_ptr[3 * triangle_idx + 2]};
    tb_float3 v1 = {attr_ptr[3 * triangle.x + 0], attr_ptr[3 * triangle.x + 1],
                    attr_ptr[3 * triangle.x + 2]};
    tb_float3 v2 = {attr_ptr[3 * triangle.y + 0], attr_ptr[3 * triangle.y + 1],
                    attr_ptr[3 * triangle.y + 2]};
    tb_float3 v3 = {attr_ptr[3 * triangle.z + 0], attr_ptr[3 * triangle.z + 1],
                    attr_ptr[3 * triangle.z + 2]};

    tb_float3 interpolated;
    interpolated.x =
        v1.x * barycentric.x + v2.x * barycentric.y + v3.x * barycentric.z;
    interpolated.y =
        v1.y * barycentric.x + v2.y * barycentric.y + v3.y * barycentric.z;
    interpolated.z =
        v1.z * barycentric.x + v2.z * barycentric.y + v3.z * barycentric.z;

    output_ptr[idx * 3 + 0] = interpolated.x;
    output_ptr[idx * 3 + 1] = interpolated.y;
    output_ptr[idx * 3 + 2] = interpolated.z;
  }

#ifdef TIMING
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Interpolation time: " << elapsed.count() << "s" << std::endl;
#endif
  return pos_bake;
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(texture_baker_cpp, m) {
  m.def("rasterize(Tensor uv, Tensor indices, int bake_resolution) -> Tensor");
  m.def("interpolate(Tensor attr, Tensor indices, Tensor rast) -> Tensor");
}

// Registers CPP implementations
TORCH_LIBRARY_IMPL(texture_baker_cpp, CPU, m) {
  m.impl("rasterize", &rasterize_cpu);
  m.impl("interpolate", &interpolate_cpu);
}

} // namespace texture_baker_cpp
