#pragma once

#include <cfloat>
#include <cmath>
#ifndef __ARM_ARCH_ISA_A64
#include <immintrin.h>
#endif
#include <limits>
#include <vector>

#include "common.h"
#include "intersect.h"
/**
 * Based on https://github.com/jbikker/bvh_article released under the unlicense.
 */

// bin count for binned BVH building
#define BINS 8

namespace UVUnwrapper {
// minimalist triangle struct
struct alignas(32) Triangle {
  uv_float2 v0;
  uv_float2 v1;
  uv_float2 v2;
  uv_float2 centroid;

  bool overlaps(const Triangle &other) {
    // return tri_tri_overlap_test_2d(v0, v1, v2, other.v0, other.v1, other.v2);
    return triangle_triangle_intersection(v0, v1, v2, other.v0, other.v1,
                                          other.v2);
  }

  bool operator==(const Triangle &rhs) const {
    return v0 == rhs.v0 && v1 == rhs.v1 && v2 == rhs.v2;
  }
};

// minimalist AABB struct with grow functionality
struct alignas(16) AABB {
  // Init bounding boxes with max/min
  uv_float2 min = {FLT_MAX, FLT_MAX};
  uv_float2 max = {FLT_MIN, FLT_MIN};

  void grow(const uv_float2 &p) {
    min.x = std::min(min.x, p.x);
    min.y = std::min(min.y, p.y);
    max.x = std::max(max.x, p.x);
    max.y = std::max(max.y, p.y);
  }

  void grow(const AABB &b) {
    if (b.min.x != FLT_MAX) {
      grow(b.min);
      grow(b.max);
    }
  }

  bool overlaps(const Triangle &tri) {
    return triangle_aabb_intersection(min, max, tri.v0, tri.v1, tri.v2);
  }

  float area() const {
    uv_float2 extent = {max.x - min.x, max.y - min.y};
    return extent.x * extent.y;
  }

  void invalidate() {
    min = {FLT_MAX, FLT_MAX};
    max = {FLT_MIN, FLT_MIN};
  }
};

// 32-byte BVH node struct
struct alignas(32) BVHNode {
  AABB bbox;              // 16
  int start = 0, end = 0; // 8
  int left, right;

  int num_triangles() const { return end - start; }

  bool is_leaf() const { return left == -1 && right == -1; }

  float calculate_node_cost() {
    float area = bbox.area();
    return num_triangles() * area;
  }
};

class BVH {
public:
  BVH() = default;
  BVH(BVH &&other) noexcept;
  BVH(const BVH &other);
  BVH &operator=(const BVH &other);
  BVH &operator=(BVH &&other) noexcept;
  BVH(Triangle *tri, int *actual_idx, const size_t &num_indices);
  ~BVH();

  std::vector<int> Intersect(Triangle &triangle);

private:
  void Subdivide(unsigned int node_idx, unsigned int &nodePtr,
                 AABB &centroidBounds);
  void UpdateNodeBounds(unsigned int nodeIdx, AABB &centroidBounds);
  float FindBestSplitPlane(BVHNode &node, int &axis, int &splitPos,
                           AABB &centroidBounds);

public:
  int *triIdx = nullptr;
  int *actualIdx = nullptr;
  unsigned int triCount;
  unsigned int nodesUsed;
  BVHNode *bvhNode = nullptr;
  Triangle *triangle = nullptr;
};

} // namespace UVUnwrapper
