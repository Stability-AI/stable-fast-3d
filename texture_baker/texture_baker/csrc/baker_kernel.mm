#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include "baker.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <filesystem>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Helper function to create a compute pipeline state object (PSO).
static inline id<MTLComputePipelineState> createComputePipelineState(id<MTLDevice> device, NSString* fullSource, std::string kernel_name) {
    NSError *error = nil;

    // Load the custom kernel shader.
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    // Add the preprocessor macro "__METAL__"
    options.preprocessorMacros = @{@"__METAL__": @""};
    id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource: fullSource options:options error:&error];
    TORCH_CHECK(customKernelLibrary, "Failed to create custom kernel library, error: ", error.localizedDescription.UTF8String);

    id<MTLFunction> customKernelFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(customKernelFunction, "Failed to create function state object for ", kernel_name.c_str());

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:customKernelFunction error:&error];
    TORCH_CHECK(pso, error.localizedDescription.UTF8String);

    return pso;
}

std::filesystem::path get_extension_path() {
    // Ensure the GIL is held before calling any Python C API function
    PyGILState_STATE gstate = PyGILState_Ensure();

    const char* module_name = "texture_baker";

    // Import the module by name
    PyObject* module = PyImport_ImportModule(module_name);
    if (!module) {
        PyGILState_Release(gstate);
        throw std::runtime_error("Could not import the module: " + std::string(module_name));
    }

    // Get the filename of the module
    PyObject* filename_obj = PyModule_GetFilenameObject(module);
    if (filename_obj) {
        std::string path = PyUnicode_AsUTF8(filename_obj);
        Py_DECREF(filename_obj);
        PyGILState_Release(gstate);

        // Get the directory part of the path (removing the __init__.py)
        std::filesystem::path module_path = std::filesystem::path(path).parent_path();

        // Append the 'csrc' directory to the path
        module_path /= "csrc";

        return module_path;
    } else {
        PyGILState_Release(gstate);
        throw std::runtime_error("Could not retrieve the module filename.");
    }
}

NSString *get_shader_sources_as_string()
{
    const std::filesystem::path csrc_path = get_extension_path();
    const std::string shader_path = (csrc_path / "baker_kernel.metal").string();
    const std::string shader_header_path = (csrc_path / "baker.h").string();
    // Load the Metal shader from the specified path
    NSError *error = nil;

    NSString* shaderHeaderSource = [
        NSString stringWithContentsOfFile:[NSString stringWithUTF8String:shader_header_path.c_str()]
        encoding:NSUTF8StringEncoding
        error:&error];
    if (error) {
        throw std::runtime_error("Failed to load baker.h: " + std::string(error.localizedDescription.UTF8String));
    }

    NSString* shaderSource = [
        NSString stringWithContentsOfFile:[NSString stringWithUTF8String:shader_path.c_str()]
        encoding:NSUTF8StringEncoding
        error:&error];
    if (error) {
        throw std::runtime_error("Failed to load Metal shader: " + std::string(error.localizedDescription.UTF8String));
    }

    NSString *fullSource = [shaderHeaderSource stringByAppendingString:shaderSource];

    return fullSource;
}

namespace texture_baker_cpp
{
    torch::Tensor rasterize_gpu(
        torch::Tensor uv,
        torch::Tensor indices,
        int64_t bake_resolution)
    {
        TORCH_CHECK(uv.device().is_mps(), "uv must be a MPS tensor");
        TORCH_CHECK(uv.is_contiguous(), "uv must be contiguous");
        TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

        TORCH_CHECK(uv.scalar_type() == torch::kFloat32, "Unsupported data type: ", indices.scalar_type());
        TORCH_CHECK(indices.scalar_type() == torch::kInt32, "Unsupported data type: ", indices.scalar_type());

        torch::Tensor rast_result = torch::empty({bake_resolution, bake_resolution, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS)).contiguous();

        @autoreleasepool {
            auto vertices_cpu = uv.contiguous().cpu();
            auto indices_cpu = indices.contiguous().cpu();

            const tb_float2 *vertices_cpu_ptr = (tb_float2*)vertices_cpu.contiguous().data_ptr<float>();
            const tb_int3 *tris_cpu_ptr = (tb_int3*)indices_cpu.contiguous().data_ptr<int>();

            BVH bvh;
            bvh.build(vertices_cpu_ptr, tris_cpu_ptr, indices.size(0));

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();

            NSString *fullSource = get_shader_sources_as_string();

            // Create a compute pipeline state object using the helper function
            id<MTLComputePipelineState> bake_uv_PSO = createComputePipelineState(device, fullSource, "kernel_bake_uv");

            // Get a reference to the command buffer for the MPS stream.
            id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
            TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

            // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
            dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

            dispatch_sync(serialQueue, ^(){
                // Start a compute pass.
                id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
                TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

                // Get Metal buffers directly from PyTorch tensors
                auto uv_buf = getMTLBufferStorage(uv.contiguous());
                auto indices_buf = getMTLBufferStorage(indices.contiguous());
                auto rast_result_buf = getMTLBufferStorage(rast_result);

                const int width = bake_resolution;
                const int height = bake_resolution;
                const int num_indices = indices.size(0);
                const int bvh_root = bvh.root;

                // Wrap the existing CPU memory in Metal buffers with shared memory
                id<MTLBuffer> nodesBuffer = [device newBufferWithBytesNoCopy:(void*)bvh.nodes.data() length:sizeof(BVHNode) * bvh.nodes.size() options:MTLResourceStorageModeShared deallocator:nil];
                id<MTLBuffer> trianglesBuffer = [device newBufferWithBytesNoCopy:(void*)bvh.triangles.data() length:sizeof(Triangle) * bvh.triangles.size() options:MTLResourceStorageModeShared deallocator:nil];
                id<MTLBuffer> triangleIndicesBuffer = [device newBufferWithBytesNoCopy:(void*)bvh.triangle_indices.data() length:sizeof(int) * bvh.triangle_indices.size() options:MTLResourceStorageModeShared deallocator:nil];

                [computeEncoder setComputePipelineState:bake_uv_PSO];
                [computeEncoder setBuffer:uv_buf offset:uv.storage_offset() * uv.element_size() atIndex:0];
                [computeEncoder setBuffer:indices_buf offset:indices.storage_offset() * indices.element_size() atIndex:1];
                [computeEncoder setBuffer:rast_result_buf offset:rast_result.storage_offset() * rast_result.element_size() atIndex:2];
                [computeEncoder setBuffer:nodesBuffer offset:0 atIndex:3];
                [computeEncoder setBuffer:trianglesBuffer offset:0 atIndex:4];
                [computeEncoder setBuffer:triangleIndicesBuffer offset:0 atIndex:5];
                [computeEncoder setBytes:&bvh_root length:sizeof(int) atIndex:6];
                [computeEncoder setBytes:&width length:sizeof(int) atIndex:7];
                [computeEncoder setBytes:&height length:sizeof(int) atIndex:8];
                [computeEncoder setBytes:&num_indices length:sizeof(int) atIndex:9];

                // Calculate a thread group size.
                int block_size = 16;
                MTLSize threadgroupSize = MTLSizeMake(block_size, block_size, 1);  // Fixed threadgroup size
                MTLSize numThreadgroups = MTLSizeMake(bake_resolution / block_size, bake_resolution / block_size, 1);

                // Encode the compute command.
                [computeEncoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];
                [computeEncoder endEncoding];

                // Commit the work.
                torch::mps::commit();
            });
        }

        return rast_result;
    }

    torch::Tensor interpolate_gpu(
        torch::Tensor attr,
        torch::Tensor indices,
        torch::Tensor rast)
    {
        TORCH_CHECK(attr.is_contiguous(), "attr must be contiguous");
        TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
        TORCH_CHECK(rast.is_contiguous(), "rast must be contiguous");

        torch::Tensor pos_bake = torch::empty({rast.size(0), rast.size(1), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS)).contiguous();
        std::filesystem::path csrc_path = get_extension_path();

        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();

            NSString *fullSource = get_shader_sources_as_string();
            // Create a compute pipeline state object using the helper function
            id<MTLComputePipelineState> interpolate_PSO = createComputePipelineState(device, fullSource, "kernel_interpolate");

            // Get a reference to the command buffer for the MPS stream.
            id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
            TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

            // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
            dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

            dispatch_sync(serialQueue, ^(){
                // Start a compute pass.
                id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
                TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

                // Get Metal buffers directly from PyTorch tensors
                auto attr_buf = getMTLBufferStorage(attr.contiguous());
                auto indices_buf = getMTLBufferStorage(indices.contiguous());
                auto rast_buf = getMTLBufferStorage(rast.contiguous());
                auto pos_bake_buf = getMTLBufferStorage(pos_bake);

                int width = rast.size(0);
                int height = rast.size(1);

                [computeEncoder setComputePipelineState:interpolate_PSO];
                [computeEncoder setBuffer:attr_buf offset:attr.storage_offset() * attr.element_size() atIndex:0];
                [computeEncoder setBuffer:indices_buf offset:indices.storage_offset() * indices.element_size() atIndex:1];
                [computeEncoder setBuffer:rast_buf offset:rast.storage_offset() * rast.element_size() atIndex:2];
                [computeEncoder setBuffer:pos_bake_buf offset:pos_bake.storage_offset() * pos_bake.element_size() atIndex:3];
                [computeEncoder setBytes:&width length:sizeof(int) atIndex:4];
                [computeEncoder setBytes:&height length:sizeof(int) atIndex:5];

                // Calculate a thread group size.

                int block_size = 16;
                MTLSize threadgroupSize = MTLSizeMake(block_size, block_size, 1);  // Fixed threadgroup size
                MTLSize numThreadgroups = MTLSizeMake(rast.size(0) / block_size, rast.size(0) / block_size, 1);

                // Encode the compute command.
                [computeEncoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];

                [computeEncoder endEncoding];

                // Commit the work.
                torch::mps::commit();
            });
        }

        return pos_bake;
    }

    // Registers MPS implementations
    TORCH_LIBRARY_IMPL(texture_baker_cpp, MPS, m)
    {
        m.impl("rasterize", &rasterize_gpu);
        m.impl("interpolate", &interpolate_gpu);
    }

}
