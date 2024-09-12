

#include "bvh.h"
#include "common.h"
#include <cstring>
#include <iostream>
#include <queue>
#include <tuple>

namespace UVUnwrapper {
BVH::BVH(Triangle *tri, int *actual_idx, const size_t &num_indices) {
  // Copty tri to triangle
  triangle = new Triangle[num_indices];
  memcpy(triangle, tri, num_indices * sizeof(Triangle));

  // Copy actual_idx to actualIdx
  actualIdx = new int[num_indices];
  memcpy(actualIdx, actual_idx, num_indices * sizeof(int));

  triIdx = new int[num_indices];
  triCount = num_indices;

  bvhNode = new BVHNode[triCount * 2 + 64];
  nodesUsed = 2;
  memset(bvhNode, 0, triCount * 2 * sizeof(BVHNode));

  // populate triangle index array
  for (int i = 0; i < triCount; i++)
    triIdx[i] = i;

  BVHNode &root = bvhNode[0];

  root.start = 0, root.end = triCount;
  AABB centroidBounds;
  UpdateNodeBounds(0, centroidBounds);

  // subdivide recursively
  Subdivide(0, nodesUsed, centroidBounds);
}

BVH::BVH(const BVH &other)
    : BVH(other.triangle, other.triIdx, other.triCount) {}

BVH::BVH(BVH &&other) noexcept // move constructor
    : triIdx(std::exchange(other.triIdx, nullptr)),
      actualIdx(std::exchange(other.actualIdx, nullptr)),
      triangle(std::exchange(other.triangle, nullptr)),
      bvhNode(std::exchange(other.bvhNode, nullptr)) {}

BVH &BVH::operator=(const BVH &other) // copy assignment
{
  return *this = BVH(other);
}

BVH &BVH::operator=(BVH &&other) noexcept // move assignment
{
  std::swap(triIdx, other.triIdx);
  std::swap(actualIdx, other.actualIdx);
  std::swap(triangle, other.triangle);
  std::swap(bvhNode, other.bvhNode);
  std::swap(triCount, other.triCount);
  std::swap(nodesUsed, other.nodesUsed);
  return *this;
}

BVH::~BVH() {
  if (triIdx)
    delete[] triIdx;
  if (triangle)
    delete[] triangle;
  if (actualIdx)
    delete[] actualIdx;
  if (bvhNode)
    delete[] bvhNode;
}

void BVH::UpdateNodeBounds(unsigned int nodeIdx, AABB &centroidBounds) {
  BVHNode &node = bvhNode[nodeIdx];
#ifndef __ARM_ARCH_ISA_A64
#ifndef _MSC_VER
  if (__builtin_cpu_supports("sse"))
#elif (defined(_M_AMD64) || defined(_M_X64))
  // SSE supported on Windows
  if constexpr (true)
#endif
  {
    __m128 min4 = _mm_set_ps1(FLT_MAX), max4 = _mm_set_ps1(FLT_MIN);
    __m128 cmin4 = _mm_set_ps1(FLT_MAX), cmax4 = _mm_set_ps1(FLT_MIN);
    for (int i = node.start; i < node.end; i += 2) {
      Triangle &leafTri1 = triangle[triIdx[i]];
      __m128 v0, v1, v2, centroid;
      if (i + 1 < node.end) {
        const Triangle leafTri2 = triangle[triIdx[i + 1]];

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
  else {
    node.bbox.invalidate();
    centroidBounds.invalidate();

    // Calculate the bounding box for the node
    for (int i = node.start; i < node.end; ++i) {
      const Triangle &tri = triangle[triIdx[i]];
      node.bbox.grow(tri.v0);
      node.bbox.grow(tri.v1);
      node.bbox.grow(tri.v2);
      centroidBounds.grow(tri.centroid);
    }
  }
}

void BVH::Subdivide(unsigned int root_idx, unsigned int &nodePtr,
                    AABB &rootCentroidBounds) {
  // Create a queue for the nodes to be subdivided
  std::queue<std::tuple<unsigned int, AABB>> nodeQueue;
  nodeQueue.push(std::make_tuple(root_idx, rootCentroidBounds));

  while (!nodeQueue.empty()) {
    // Get the next node to process from the queue
    auto [node_idx, centroidBounds] = nodeQueue.front();
    nodeQueue.pop();
    BVHNode &node = bvhNode[node_idx];

    // Check if left is -1 and right not or vice versa

    int axis, splitPos;
    float cost = FindBestSplitPlane(node, axis, splitPos, centroidBounds);

    if (cost >= node.calculate_node_cost()) {
      node.left = node.right = -1;
      continue; // Move on to the next node in the queue
    }

    int i = node.start;
    int j = node.end - 1;
    float scale = BINS / (centroidBounds.max[axis] - centroidBounds.min[axis]);
    while (i <= j) {
      int binIdx =
          std::min(BINS - 1, (int)((triangle[triIdx[i]].centroid[axis] -
                                    centroidBounds.min[axis]) *
                                   scale));
      if (binIdx < splitPos)
        i++;
      else
        std::swap(triIdx[i], triIdx[j--]);
    }

    int leftCount = i - node.start;
    if (leftCount == 0 || leftCount == (int)node.num_triangles()) {
      node.left = node.right = -1;
      continue; // Move on to the next node in the queue
    }

    int mid = i;

    // Create child nodes
    int leftChildIdx = nodePtr++;
    int rightChildIdx = nodePtr++;
    bvhNode[leftChildIdx].start = node.start;
    bvhNode[leftChildIdx].end = mid;
    bvhNode[rightChildIdx].start = mid;
    bvhNode[rightChildIdx].end = node.end;
    node.left = leftChildIdx;
    node.right = rightChildIdx;

    // Update the bounds for the child nodes and push them onto the queue
    UpdateNodeBounds(leftChildIdx, centroidBounds);
    nodeQueue.push(std::make_tuple(leftChildIdx, centroidBounds));

    UpdateNodeBounds(rightChildIdx, centroidBounds);
    nodeQueue.push(std::make_tuple(rightChildIdx, centroidBounds));
  }
}

float BVH::FindBestSplitPlane(BVHNode &node, int &best_axis, int &best_pos,
                              AABB &centroidBounds) {
  float best_cost = FLT_MAX;

  for (int axis = 0; axis < 2; ++axis) // We use 2 as we have only x and y
  {
    float boundsMin = centroidBounds.min[axis];
    float boundsMax = centroidBounds.max[axis];
    // Or floating point precision
    if ((boundsMin == boundsMax) || (boundsMax - boundsMin < 1e-8f)) {
      continue;
    }

    // populate the bins
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
        min4[i] = _mm_set_ps1(FLT_MAX), max4[i] = _mm_set_ps1(FLT_MIN),
        count[i] = 0;
      for (int i = node.start; i < node.end; i++) {
        Triangle &tri = triangle[triIdx[i]];
        int binIdx =
            std::min(BINS - 1, (int)((tri.centroid[axis] - boundsMin) * scale));
        count[binIdx]++;

        __m128 v0 = _mm_set_ps(tri.v0.x, tri.v0.y, 0.0f, 0.0f);
        __m128 v1 = _mm_set_ps(tri.v1.x, tri.v1.y, 0.0f, 0.0f);
        __m128 v2 = _mm_set_ps(tri.v2.x, tri.v2.y, 0.0f, 0.0f);
        min4[binIdx] = _mm_min_ps(min4[binIdx], v0);
        max4[binIdx] = _mm_max_ps(max4[binIdx], v0);
        min4[binIdx] = _mm_min_ps(min4[binIdx], v1);
        max4[binIdx] = _mm_max_ps(max4[binIdx], v1);
        min4[binIdx] = _mm_min_ps(min4[binIdx], v2);
        max4[binIdx] = _mm_max_ps(max4[binIdx], v2);
      }
      // gather data for the 7 planes between the 8 bins
      __m128 leftMin4 = _mm_set_ps1(FLT_MAX), rightMin4 = leftMin4;
      __m128 leftMax4 = _mm_set_ps1(FLT_MIN), rightMax4 = leftMax4;
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
      } bin[BINS];
      for (int i = node.start; i < node.end; i++) {
        Triangle &tri = triangle[triIdx[i]];
        int binIdx =
            std::min(BINS - 1, (int)((tri.centroid[axis] - boundsMin) * scale));
        bin[binIdx].triCount++;
        bin[binIdx].bounds.grow(tri.v0);
        bin[binIdx].bounds.grow(tri.v1);
        bin[binIdx].bounds.grow(tri.v2);
      }
      // gather data for the 7 planes between the 8 bins
      AABB leftBox, rightBox;
      for (int i = 0; i < BINS - 1; i++) {
        leftSum += bin[i].triCount;
        leftBox.grow(bin[i].bounds);
        leftCountArea[i] = leftSum * leftBox.area();
        rightSum += bin[BINS - 1 - i].triCount;
        rightBox.grow(bin[BINS - 1 - i].bounds);
        rightCountArea[BINS - 2 - i] = rightSum * rightBox.area();
      }
    }

    // calculate SAH cost for the 7 planes
    scale = (boundsMax - boundsMin) / BINS;
    for (int i = 0; i < BINS - 1; i++) {
      const float planeCost = leftCountArea[i] + rightCountArea[i];
      if (planeCost < best_cost)
        best_axis = axis, best_pos = i + 1, best_cost = planeCost;
    }
  }
  return best_cost;
}

std::vector<int> BVH::Intersect(Triangle &tri_intersect) {
  /**
   * @brief Intersect a triangle with the BVH
   *
   * @param triangle the triangle to intersect
   *
   * @return -1 for no intersection, the index of the intersected triangle
   * otherwise
   */

  const int max_stack_size = 64;
  int node_stack[max_stack_size];
  int stack_size = 0;
  std::vector<int> intersected_triangles;

  node_stack[stack_size++] = 0; // Start with the root node (index 0)
  while (stack_size > 0) {
    int node_idx = node_stack[--stack_size];
    const BVHNode &node = bvhNode[node_idx];
    if (node.is_leaf()) {
      for (int i = node.start; i < node.end; ++i) {
        const Triangle &tri = triangle[triIdx[i]];
        // Check that the triangle is not the same as the intersected triangle
        if (tri == tri_intersect)
          continue;
        if (tri_intersect.overlaps(tri)) {
          intersected_triangles.push_back(actualIdx[triIdx[i]]);
        }
      }
    } else {
      // Check right child first
      if (bvhNode[node.right].bbox.overlaps(tri_intersect)) {
        if (stack_size < max_stack_size) {
          node_stack[stack_size++] = node.right;
        } else {
          throw std::runtime_error("Node stack overflow");
        }
      }

      // Check left child
      if (bvhNode[node.left].bbox.overlaps(tri_intersect)) {
        if (stack_size < max_stack_size) {
          node_stack[stack_size++] = node.left;
        } else {
          throw std::runtime_error("Node stack overflow");
        }
      }
    }
  }
  return intersected_triangles; // Return all intersected triangle indices
}

} // namespace UVUnwrapper
