// CSR is a sparse (Mostly zero) matrix representation made from 3 arrays
// Array 1 is the rowPtrs, Array 2 is the colIdx, and Array 3 is the value
struct CSRMatrix {
    int numNonZeros;
    unsigned int* rowPtrs;
    unsigned int* colIdx;
    float* value;
}


// This kernel performs a sparse matric vector (spmv) multiplication between csrMatrix and x and outputs the results to y
__global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < csrMatrix.numNonZeros) {
        float sum = 0.0f;
        for (unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row]; i++) {
            unsigned int col = csrMatrix.colIdx[i];
            float val = csrMatrix.value[i];
            sum += x[col]*val;
        }
        y[row] = sum;
    }
}