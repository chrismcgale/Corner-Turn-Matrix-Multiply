#include "spmv_coo.cu"
#include "spmv_csr.cu"

// Arbitrary, tune to data
#DEFINE SECTION_SIZE 128;

__host__ CSRMatrix convert_coo_csr(COOMatrix cooMatrix) {
        // Use num of rows possible, we'll remove excess them in the scan
    unsigned int length = sizeof(cooMatrix.rowIdx) / sizeof(cooMatrix.rowIdx[0]);
    unsigned int histo[length];
    unsigned int filtered[length];

    cudaMalloc((void**)&cooMatrix.rowIdx, length * sizeof(int));
    cudaMalloc((void**)&histo, length * sizeof(int));
    
    // Arbitrary dims, fine tuning needed
    parallel_histogram_kernel<<<(length / 128), 128>>>(cooMatrix.rowIdx, length, histo);

    
    cudaFree(cooMatrix.rowIdx);
    cudaFree(histo);


    int j = 0;
    for (i = 0; i < length; i++) {
        if (histo[i] != 0) { // Filtering condition (example: keeping even numbers)
            filtered[j] = histo[i];
            j++;
        }
    }

    unsigned int summed[j];

    cudaMalloc((void**)&filtered, j * sizeof(int));
    cudaMalloc((void**)&summed, j * sizeof(int));

    Kogge_Stone_scan_kernel<<<(j / 128), 128>>>(filtered, summed, j);

    cudaFree(filtered);
    cudaFree(summed);

    return {numNonZeros, summed, cooMatrix.colIdx, cooMatrix.value};
}


// Basic parallel histogram
__global__ void parallel_histogram_kernel(unsigned int *data, unsigned int length, unsigned int *histo) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < length) {
        int rowIdx = data[i];
        if (rowIdx >= 0) {
            // histo is shared between threads so we need atomicity
            atomicAdd(&(histo[rowIdx]), 1);
        }
    }
}

// Parallel prefix sum algorithm
__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int n, bool inclusive = true) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n || (!inclusive && threadIdx.x = 0)) {
        if (inclusive) {
            XY[threadIdx.x] = X[i];
        } else {
            XY[threadIdx.x] = X[i - 1];
        }
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            XY[threadIdx.x] = temp;
        }
    }
    if (i < n) {
        Y[i] = XY[threadIdx.x];
    }
}