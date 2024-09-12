#include "bvh.h"
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <set>
#include <torch/extension.h>
#include <vector>

// #define TIMING

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace UVUnwrapper {
void create_bvhs(BVH *bvhs, Triangle *triangles,
                 std::vector<std::set<int>> &triangle_per_face, int num_faces,
                 int start, int end) {
#pragma omp parallel for
  for (int i = start; i < end; i++) {
    int num_triangles = triangle_per_face[i].size();
    Triangle *triangles_per_face = new Triangle[num_triangles];
    int *indices = new int[num_triangles];
    int j = 0;
    for (int idx : triangle_per_face[i]) {
      triangles_per_face[j] = triangles[idx];
      indices[j++] = idx;
    }
    // Each thread writes to it's own memory space
    // First check if the number of triangles is 0
    if (num_triangles == 0) {
      bvhs[i - start] = std::move(BVH()); // Default constructor
    } else {
      bvhs[i - start] = std::move(
          BVH(triangles_per_face, indices,
              num_triangles)); // BVH now handles memory of triangles_per_face
    }
    delete[] triangles_per_face;
  }
}

void perform_intersection_check(BVH *bvhs, int num_bvhs, Triangle *triangles,
                                uv_float3 *vertex_tri_centroids,
                                int64_t *assign_indices_ptr,
                                ssize_t num_indices, int offset,
                                std::vector<std::set<int>> &triangle_per_face) {
  std::vector<std::pair<int, int>>
      unique_intersections; // Store unique intersections as pairs of triangle
                            // indices

// Step 1: Detect intersections in parallel
#pragma omp parallel for
  for (int i = 0; i < num_indices; i++) {
    if (assign_indices_ptr[i] < offset) {
      continue;
    }

    Triangle cur_tri = triangles[i];
    auto &cur_bvh = bvhs[assign_indices_ptr[i] - offset];

    if (cur_bvh.bvhNode == nullptr) {
      continue;
    }

    std::vector<int> intersections = cur_bvh.Intersect(cur_tri);

    if (!intersections.empty()) {

#pragma omp critical
      {
        for (int intersect : intersections) {
          if (i != intersect) {
            // Ensure we only store unique pairs (A, B) where A < B to avoid
            // duplication
            if (i < intersect) {
              unique_intersections.push_back(std::make_pair(i, intersect));
            } else {
              unique_intersections.push_back(std::make_pair(intersect, i));
            }
          }
        }
      }
    }
  }

  // Step 2: Process unique intersections
  for (int idx = 0; idx < unique_intersections.size(); idx++) {
    int first = unique_intersections[idx].first;
    int second = unique_intersections[idx].second;

    int i_idx = assign_indices_ptr[first];

    int norm_idx = i_idx % 6;
    int axis = (norm_idx < 2) ? 0 : (norm_idx < 4) ? 1 : 2;
    bool use_max = (i_idx % 2) == 1;

    float pos_a = vertex_tri_centroids[first][axis];
    float pos_b = vertex_tri_centroids[second][axis];
    // Sort the intersections based on vertex_tri_centroids along the specified
    // axis
    if (use_max) {
      if (pos_a < pos_b) {
        std::swap(first, second);
      }
    } else {
      if (pos_a > pos_b) {
        std::swap(first, second);
      }
    }

    // Update the unique intersections
    unique_intersections[idx].first = first;
    unique_intersections[idx].second = second;
  }

  // Now only get the second intersections from the pair and put them in a set
  // The second intersection should always be the occluded triangle
  std::set<int> second_intersections;
  for (int idx = 0; idx < (int)unique_intersections.size(); idx++) {
    int second = unique_intersections[idx].second;
    second_intersections.insert(second);
  }

  for (int int_idx : second_intersections) {
    // Move the second (occluded) triangle by 6
    int intersect_idx = assign_indices_ptr[int_idx];
    int new_index = intersect_idx + 6;
    new_index = std::clamp(new_index, 0, 12);

    assign_indices_ptr[int_idx] = new_index;
    triangle_per_face[intersect_idx].erase(int_idx);
    triangle_per_face[new_index].insert(int_idx);
  }
}

torch::Tensor assign_faces_uv_to_atlas_index(torch::Tensor vertices,
                                             torch::Tensor indices,
                                             torch::Tensor face_uv,
                                             torch::Tensor face_index) {
  // Get the number of faces
  int num_faces = indices.size(0);
  torch::Tensor assign_indices =
      torch::empty(
          {
              num_faces,
          },
          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
          .contiguous();

  auto vert_accessor = vertices.accessor<float, 2>();
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto face_uv_accessor = face_uv.accessor<float, 2>();

  const int64_t *face_index_ptr = face_index.contiguous().data_ptr<int64_t>();
  int64_t *assign_indices_ptr = assign_indices.data_ptr<int64_t>();
  // copy face_index to assign_indices
  memcpy(assign_indices_ptr, face_index_ptr, num_faces * sizeof(int64_t));

#ifdef TIMING
  auto start = std::chrono::high_resolution_clock::now();
#endif
  uv_float3 *vertex_tri_centroids = new uv_float3[num_faces];
  Triangle *triangles = new Triangle[num_faces];

  // Use std::set to store triangles for each face
  std::vector<std::set<int>> triangle_per_face;
  triangle_per_face.resize(13);

#pragma omp parallel for
  for (int i = 0; i < num_faces; i++) {
    int face_idx = i * 3;
    triangles[i].v0 = {face_uv_accessor[face_idx + 0][0],
                       face_uv_accessor[face_idx + 0][1]};
    triangles[i].v1 = {face_uv_accessor[face_idx + 1][0],
                       face_uv_accessor[face_idx + 1][1]};
    triangles[i].v2 = {face_uv_accessor[face_idx + 2][0],
                       face_uv_accessor[face_idx + 2][1]};
    triangles[i].centroid =
        triangle_centroid(triangles[i].v0, triangles[i].v1, triangles[i].v2);

    uv_float3 v0 = {vert_accessor[indices_accessor[i][0]][0],
                    vert_accessor[indices_accessor[i][0]][1],
                    vert_accessor[indices_accessor[i][0]][2]};
    uv_float3 v1 = {vert_accessor[indices_accessor[i][1]][0],
                    vert_accessor[indices_accessor[i][1]][1],
                    vert_accessor[indices_accessor[i][1]][2]};
    uv_float3 v2 = {vert_accessor[indices_accessor[i][2]][0],
                    vert_accessor[indices_accessor[i][2]][1],
                    vert_accessor[indices_accessor[i][2]][2]};
    vertex_tri_centroids[i] = triangle_centroid(v0, v1, v2);

// Assign the triangle to the face index
#pragma omp critical
    { triangle_per_face[face_index_ptr[i]].insert(i); }
  }

#ifdef TIMING
  auto start_bvh = std::chrono::high_resolution_clock::now();
#endif

  BVH *bvhs = new BVH[6];
  create_bvhs(bvhs, triangles, triangle_per_face, num_faces, 0, 6);

#ifdef TIMING
  auto end_bvh = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_bvh - start_bvh;
  std::cout << "BVH build time: " << elapsed_seconds.count() << "s\n";

  auto start_intersection_1 = std::chrono::high_resolution_clock::now();
#endif

  perform_intersection_check(bvhs, 6, triangles, vertex_tri_centroids,
                             assign_indices_ptr, num_faces, 0,
                             triangle_per_face);

#ifdef TIMING
  auto end_intersection_1 = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end_intersection_1 - start_intersection_1;
  std::cout << "Intersection 1 time: " << elapsed_seconds.count() << "s\n";
#endif
  // Create 6 new bvhs and delete the old ones
  BVH *new_bvhs = new BVH[6];
  create_bvhs(new_bvhs, triangles, triangle_per_face, num_faces, 6, 12);

#ifdef TIMING
  auto end_bvh2 = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end_bvh2 - end_intersection_1;
  std::cout << "BVH 2 build time: " << elapsed_seconds.count() << "s\n";
  auto start_intersection_2 = std::chrono::high_resolution_clock::now();
#endif

  perform_intersection_check(new_bvhs, 6, triangles, vertex_tri_centroids,
                             assign_indices_ptr, num_faces, 6,
                             triangle_per_face);

#ifdef TIMING
  auto end_intersection_2 = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end_intersection_2 - start_intersection_2;
  std::cout << "Intersection 2 time: " << elapsed_seconds.count() << "s\n";
  elapsed_seconds = end_intersection_2 - start;
  std::cout << "Total time: " << elapsed_seconds.count() << "s\n";
#endif

  // Cleanup
  delete[] vertex_tri_centroids;
  delete[] triangles;
  delete[] bvhs;
  delete[] new_bvhs;

  return assign_indices;
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(UVUnwrapper, m) {
  m.def("assign_faces_uv_to_atlas_index(Tensor vertices, Tensor indices, "
        "Tensor face_uv, Tensor face_index) -> Tensor");
}

// Registers CPP implementations
TORCH_LIBRARY_IMPL(UVUnwrapper, CPU, m) {
  m.impl("assign_faces_uv_to_atlas_index", &assign_faces_uv_to_atlas_index);
}

} // namespace UVUnwrapper
