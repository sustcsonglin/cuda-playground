#include <torch/types.h>



__global__ void parallel_scan_kernel(float* d_out, const float* d_in, int B, int D, int L) {
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;
    int laneId = tid & 31;
    int warpId = tid >> 5;

    int b = blockIdx.x;
    int d = blockIdx.y;

    int global_idx = ((b * D + d) * L) + tid;
    
    float value = d_in[global_idx];
    
    // Compute inclusive scan within warp
    for (int offset = 1; offset < 32; offset *= 2) {
        float up_value = __shfl_up_sync(0xffffffff, value, offset);
        if (laneId >= offset) {
            value += up_value;
        }
    }

    // The last thread of each warp writes its result to shared memory
    if (laneId == 31) {
        warp_sums[warpId] = value;
    }

    __syncthreads();

    // Use the first warp to compute the scan of the warp_sums
    if (warpId == 0) {
        float acc_value = warp_sums[laneId];
        
    
        for (int offset = 1; offset < 32; offset *= 2) {
            float up_value = __shfl_up_sync(0xffffffff, acc_value, offset);
            if (laneId >= offset) {
            acc_value += up_value;
            }
        }

        warp_sums[laneId] = acc_value;
    }

    __syncthreads();

    // Add the scanned sum of the previous warps to the current warp's result
    if (warpId != 0) {
        value += warp_sums[warpId - 1];
    }

    // Store result
    d_out[tid + ((b * D + d) * L)] = value;
}



void parallel_scan(torch::Tensor d_out, torch::Tensor d_in, int B, int D, int L) {
    const dim3 blockSize(1024, 1, 1);
    const dim3 gridSize(B, D, 1);
    
    parallel_scan_kernel<<<gridSize, blockSize>>>(d_out.data_ptr<float>(), d_in.data_ptr<float>(), B, D, L);
}



