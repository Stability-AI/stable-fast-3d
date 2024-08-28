#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "baker.h"

// #define TIMING

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CUDA_CHECK_THROW(x) \
	do { \
		cudaError_t _result = x; \
		if (_result != cudaSuccess) \
			throw std::runtime_error(std::string(FILE_LINE " check failed " #x " failed: ") + cudaGetErrorString(_result)); \
	} while(0)

namespace texture_baker_cpp
{

    __device__ float3 operator+(const float3 &a, const float3 &b)
    {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    // xy: 2D test position
    // v1: vertex position 1
    // v2: vertex position 2
    // v3: vertex position 3
    //
    __forceinline__ __device__ bool barycentric_coordinates(const float2 &xy, const tb_float2 &v1, const tb_float2 &v2, const tb_float2 &v3, float &u, float &v, float &w)
    {
        // Return true if the point (xy) is inside the triangle defined by the vertices v1, v2, v3.
        // If the point is inside the triangle, the barycentric coordinates are stored in u, v, and w.
        float2 v1v2 = make_float2(v2.x - v1.x, v2.y - v1.y);
        float2 v1v3 = make_float2(v3.x - v1.x, v3.y - v1.y);
        float2 xyv1 = make_float2(xy.x - v1.x, xy.y - v1.y);

        float d00 = v1v2.x * v1v2.x + v1v2.y * v1v2.y;
        float d01 = v1v2.x * v1v3.x + v1v2.y * v1v3.y;
        float d11 = v1v3.x * v1v3.x + v1v3.y * v1v3.y;
        float d20 = xyv1.x * v1v2.x + xyv1.y * v1v2.y;
        float d21 = xyv1.x * v1v3.x + xyv1.y * v1v3.y;

        float denom = d00 * d11 - d01 * d01;
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0f - v - w;

        return (v >= 0.0f) && (w >= 0.0f) && (v + w <= 1.0f);
    }

    __global__ void kernel_interpolate(const float3* __restrict__ attr, const int3* __restrict__ indices, const float4* __restrict__ rast, float3* __restrict__ output, int width, int height)
    {
        // Interpolate the attr into output based on the rast result (barycentric coordinates, + triangle idx)
        //int idx = x * width + y;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int x = idx / width;
        int y = idx % width;

        if (x >= width || y >= height)
            return;

        float4 barycentric = rast[idx];
        int triangle_idx = int(barycentric.w);

        if (triangle_idx < 0)
        {
            output[idx] = make_float3(0.0f, 0.0f, 0.0f);
            return;
        }

        float3 v1 = attr[indices[triangle_idx].x];
        float3 v2 = attr[indices[triangle_idx].y];
        float3 v3 = attr[indices[triangle_idx].z];

        output[idx] = make_float3(v1.x * barycentric.x, v1.y * barycentric.x, v1.z * barycentric.x)
        + make_float3(v2.x * barycentric.y, v2.y * barycentric.y, v2.z * barycentric.y)
        + make_float3(v3.x * barycentric.z, v3.y * barycentric.z, v3.z * barycentric.z);
    }

    __device__ bool bvh_intersect(
        const BVHNode* __restrict__ nodes,
        const Triangle* __restrict__ triangles,
        const int* __restrict__ triangle_indices,
        const int root,
        const float2 &point,
        float &u, float &v, float &w,
        int &index)
    {
        constexpr int max_stack_size = 64;
        int node_stack[max_stack_size];
        int stack_size = 0;

        node_stack[stack_size++] = root;

        while (stack_size > 0)
        {
            int node_idx = node_stack[--stack_size];
            const BVHNode &node = nodes[node_idx];

            if (node.is_leaf())
            {
                for (int i = node.start; i < node.end; ++i)
                {
                    const Triangle &tri = triangles[triangle_indices[i]];
                    if (barycentric_coordinates(point, tri.v0, tri.v1, tri.v2, u, v, w))
                    {
                        index = tri.index;
                        return true;
                    }
                }
            }
            else
            {
                if (nodes[node.right].bbox.overlaps(point))
                {
                    if (stack_size < max_stack_size)
                    {
                        node_stack[stack_size++] = node.right;
                    }
                    else
                    {
                        // Handle stack overflow
                        // Make sure NDEBUG is not defined (see setup.py)
                        assert(0 && "Node stack overflow");
                    }
                }
                if (nodes[node.left].bbox.overlaps(point))
                {
                    if (stack_size < max_stack_size)
                    {
                        node_stack[stack_size++] = node.left;
                    }
                    else
                    {
                        // Handle stack overflow
                        // Make sure NDEBUG is not defined (see setup.py)
                        assert(0 && "Node stack overflow");
                    }
                }
            }
        }

        return false;
    }

    __global__ void kernel_bake_uv(
        float2* __restrict__ uv,
        int3* __restrict__ indices,
        float4* __restrict__ output,
        const BVHNode* __restrict__ nodes,
        const Triangle* __restrict__ triangles,
        const int* __restrict__ triangle_indices,
        const int root,
        const int width,
        const int height,
        const int num_indices)
    {
        //int idx = x * width + y;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int x = idx / width;
        int y = idx % width;

        if (y >= width || x >= height)
            return;

        // We index x,y but the original coords are HW. So swap them
        float2 pixel_coord = make_float2(float(y) / height, float(x) / width);
        pixel_coord.x = fminf(fmaxf(pixel_coord.x, 0.0f), 1.0f);
        pixel_coord.y = 1.0f - fminf(fmaxf(pixel_coord.y, 0.0f), 1.0f);

        float u, v, w;
        int triangle_idx;
        bool hit = bvh_intersect(nodes, triangles, triangle_indices, root, pixel_coord, u, v, w, triangle_idx);

        if (hit)
        {
            output[idx] = make_float4(u, v, w, float(triangle_idx));
            return;
        }

        output[idx] = make_float4(0.0f, 0.0f, 0.0f, -1.0f);
    }

    torch::Tensor rasterize_gpu(
        torch::Tensor uv,
        torch::Tensor indices,
        int64_t bake_resolution)
    {
#ifdef TIMING
        auto start = std::chrono::high_resolution_clock::now();
#endif
        constexpr int block_size = 16 * 16;
        int grid_size = bake_resolution * bake_resolution / block_size;
        dim3 block_dims(block_size, 1, 1);
        dim3 grid_dims(grid_size, 1, 1);

        int num_indices = indices.size(0);

        int width = bake_resolution;
        int height = bake_resolution;

        // Step 1: create an empty tensor to store the output.
        torch::Tensor rast_result = torch::empty({bake_resolution, bake_resolution, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto vertices_cpu = uv.contiguous().cpu();
        auto indices_cpu = indices.contiguous().cpu();

        const tb_float2 *vertices_cpu_ptr = (tb_float2*)vertices_cpu.contiguous().data_ptr<float>();
        const tb_int3 *tris_cpu_ptr = (tb_int3*)indices_cpu.contiguous().data_ptr<int>();

        BVH bvh;
        bvh.build(vertices_cpu_ptr, tris_cpu_ptr, indices.size(0));

        BVHNode *nodes_gpu = nullptr;
        Triangle *triangles_gpu = nullptr;
        int *triangle_indices_gpu = nullptr;
        const int bvh_root = bvh.root;
        cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

        CUDA_CHECK_THROW(cudaMallocAsync(&nodes_gpu, sizeof(BVHNode) * bvh.nodes.size(), cuda_stream));
        CUDA_CHECK_THROW(cudaMallocAsync(&triangles_gpu, sizeof(Triangle) * bvh.triangles.size(), cuda_stream));
        CUDA_CHECK_THROW(cudaMallocAsync(&triangle_indices_gpu, sizeof(int) * bvh.triangle_indices.size(), cuda_stream));

        CUDA_CHECK_THROW(cudaMemcpyAsync(nodes_gpu, bvh.nodes.data(), sizeof(BVHNode) * bvh.nodes.size(), cudaMemcpyHostToDevice, cuda_stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(triangles_gpu, bvh.triangles.data(), sizeof(Triangle) * bvh.triangles.size(), cudaMemcpyHostToDevice, cuda_stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(triangle_indices_gpu, bvh.triangle_indices.data(), sizeof(int) * bvh.triangle_indices.size(), cudaMemcpyHostToDevice, cuda_stream));

        kernel_bake_uv<<<grid_dims, block_dims, 0, cuda_stream>>>(
            (float2 *)uv.contiguous().data_ptr<float>(),
            (int3 *)indices.contiguous().data_ptr<int>(),
            (float4 *)rast_result.contiguous().data_ptr<float>(),
            nodes_gpu,
            triangles_gpu,
            triangle_indices_gpu,
            bvh_root,
            width,
            height,
            num_indices);

        CUDA_CHECK_THROW(cudaFreeAsync(nodes_gpu, cuda_stream));
        CUDA_CHECK_THROW(cudaFreeAsync(triangles_gpu, cuda_stream));
        CUDA_CHECK_THROW(cudaFreeAsync(triangle_indices_gpu, cuda_stream));

#ifdef TIMING
        CUDA_CHECK_THROW(cudaStreamSynchronize(cuda_stream));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Rasterization time (CUDA): " << elapsed.count() << "s" << std::endl;
#endif
        return rast_result;
    }

    torch::Tensor interpolate_gpu(
        torch::Tensor attr,
        torch::Tensor indices,
        torch::Tensor rast)
    {
#ifdef TIMING
        auto start = std::chrono::high_resolution_clock::now();
#endif
        constexpr int block_size = 16 * 16;
        int grid_size = rast.size(0) * rast.size(0) / block_size;
        dim3 block_dims(block_size, 1, 1);
        dim3 grid_dims(grid_size, 1, 1);

        // Step 1: create an empty tensor to store the output.
        torch::Tensor pos_bake = torch::empty({rast.size(0), rast.size(1), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        int width = rast.size(0);
        int height = rast.size(1);

        cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

        kernel_interpolate<<<grid_dims, block_dims, 0, cuda_stream>>>(
            (float3 *)attr.contiguous().data_ptr<float>(),
            (int3 *)indices.contiguous().data_ptr<int>(),
            (float4 *)rast.contiguous().data_ptr<float>(),
            (float3 *)pos_bake.contiguous().data_ptr<float>(),
            width,
            height);
#ifdef TIMING
        CUDA_CHECK_THROW(cudaStreamSynchronize(cuda_stream));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Interpolation time (CUDA): " << elapsed.count() << "s" << std::endl;
#endif
        return pos_bake;
    }

    // Registers CUDA implementations
    TORCH_LIBRARY_IMPL(texture_baker_cpp, CUDA, m)
    {
        m.impl("rasterize", &rasterize_gpu);
        m.impl("interpolate", &interpolate_gpu);
    }

}
