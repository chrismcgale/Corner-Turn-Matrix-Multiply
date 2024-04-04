// Sparse matrix format known as Jagged Diagonal Storage (JDS) Meant to improve coallescing memory access and control divergence
struct JDSMatrix {
    unsigned int* originalRows;
    unsigned int* nnzPerRow;
    unsigned int* iterPtr;
    unsigned int* colIdx;
    unsigned int* value;
}

// This kernel performs a sparse matric vector (spmv) multiplication between jdsMatrix and x and outputs the results to y
__global__ void spmv_jds_kernel(JDSMatrix jdsMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < jdsMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < jdsMatrix.nnzPerRow[row]; t++) {
            unsigned int i = t*jdsMatrix.numRows + row;
            unsigned int col = jdsMatrix.colIdx[col];
            float value = jdsMatrix.value[i];
            sum += x[col]*value;
        }
        y[row] = sum;
    }
}