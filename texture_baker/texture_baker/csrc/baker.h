#pragma once

#include <vector>
#include <cmath>

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__METAL__)
#define CUDA_ENABLED
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace texture_baker_cpp
{
    // Structure to represent a 2D point or vector
    union alignas(8) tb_float2
    {
        struct
        {
            float x,y;
        };

        float data[2];

        float &operator[](size_t idx)
        {
            if(idx > 1)
                throw std::runtime_error("bad index");
            return data[idx];
        }

        const float &operator[](size_t idx) const
        {
            if(idx > 1)
                throw std::runtime_error("bad index");
            return data[idx];
        }

        bool operator==(const tb_float2 &rhs) const
        {
            return x == rhs.x && y == rhs.y;
        }
    };

    union alignas(16) tb_float4
    {
        struct
        {
            float x, y, z, w;
        };

        float data[4];

        float &operator[](size_t idx)
        {
            if(idx > 3)
                throw std::runtime_error("bad index");
            return data[idx];
        }

        const float &operator[](size_t idx) const
        {
            if(idx > 3)
                throw std::runtime_error("bad index");
            return data[idx];
        }
    };

    union alignas(4) tb_int3
    {
        struct
        {
            int x, y, z;
        };

        int data[3];

        int &operator[](size_t idx)
        {
            if(idx > 2)
                throw std::runtime_error("bad index");
            return data[idx];
        }
    };

    // BVH structure to accelerate point-triangle intersection
    struct alignas(16) AABB
    {
        // Init bounding boxes with max/min
        tb_float2 min = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        tb_float2 max = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};

        // grow the AABB to include a point
        void grow(const tb_float2 &p)
        {
            min.x = std::min(min.x, p.x);
            min.y = std::min(min.y, p.y);
            max.x = std::max(max.x, p.x);
            max.y = std::max(max.y, p.y);
        }

        void grow(const AABB &b)
        {
            if (b.min.x != std::numeric_limits<float>::max())
            {
                grow(b.min);
                grow(b.max);
            }
        }

        // Check if two AABBs overlap
        bool overlaps(const AABB &other) const
        {
            return min.x <= other.max.x && max.x >= other.min.x &&
                   min.y <= other.max.y && max.y >= other.min.y;
        }

        bool overlaps(const tb_float2 &point) const
        {
            return point.x >= min.x && point.x <= max.x &&
                   point.y >= min.y && point.y <= max.y;
        }

#ifdef CUDA_ENABLED
        __device__ bool overlaps(const float2 &point) const
        {
            return point.x >= min.x && point.x <= max.x &&
                   point.y >= min.y && point.y <= max.y;
        }
#endif

        // Initialize AABB to an invalid state
        void invalidate()
        {
            min = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            max = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
        }

        // Calculate the area of the AABB
        float area() const
        {
            float2 extent = {max.x - min.x, max.y - min.y};
            return extent.x * extent.y;
        }
    };

    struct BVHNode
    {
        AABB bbox;
        int start, end;
        int left, right;

        int num_triangles() const
        {
            return end - start;
        }

        CUDA_HOST_DEVICE bool is_leaf() const
        {
            return left == -1 && right == -1;
        }

        float calculate_node_cost()
        {
            float area = bbox.area();
            return num_triangles() * area;
        }
    };

    struct Triangle
    {
        tb_float2 v0, v1, v2;
        int index;
        tb_float2 centroid;
    };

    struct BVH
    {
        std::vector<BVHNode> nodes;
        std::vector<Triangle> triangles;
        std::vector<int> triangle_indices;
        int root;

        void build(const tb_float2 *vertices, const tb_int3 *indices, const int64_t &num_indices);
        bool intersect(const tb_float2 &point, float &u, float &v, float &w, int &index) const;

        void update_node_bounds(BVHNode &node, AABB &centroidBounds);
        float find_best_split_plane(const BVHNode &node, int &best_axis, int &best_pos, AABB &centroidBounds);
    };


}
