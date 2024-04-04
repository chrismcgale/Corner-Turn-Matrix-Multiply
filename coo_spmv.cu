// COO is a sparse (Mostly zero) matrix representation made from 3 arrays
// Array 1 is the rowIdx, Array 2 is the colIdx, and Array 3 is the value
struct COOMatrix {
    int numNonZeros;
    unsigned int* rowIdx;
    unsigned int* colIdx;
    float* value;
}


// This kernel performs a sparse matric vector (spmv) multiplication between cooMatrix and x and outputs the results to y
__global__ void spmv_coo_kernel(COOMatrix cooMatrix, float* x, float* y) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < cooMatrix.numNonZeros) {
        unsigned int row = cooMatrix.rowIdx[i];
        unsigned int col = cooMatrix.colIdx[i];
        float val = cooMatrix.value[i];

        // Major downside of COO is that it requires this atomicAdd
        atomicAdd(&y[row], x[col]*val);
    }
}