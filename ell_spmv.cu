// ELL is a sparse (Mostly zero) matrix representation made from 3 arrays
struct ELLMatrix {
    int numNonZeros;
    unsigned int* nnzPerRow;
    unsigned int* colIdx;
    float* value;
}


__host__ convert_csr_ell(CSRMatrix csr) {
    // Find max number of non-zero items in any row
    unsigned int max_nz_items = 0;
    unsigned int num_rows = sizeof(csr.rowPtrs) / sizeof(csr.rowPtrs[0]);
    unsigned int num_cols = sizeof(csr.colIdx) / sizeof(csr.colIdx[0]);
    unsigned int nnzPerRow[num_rows] = {0};


    unsigned int last = 0;

    for (unsigned int i = 0; i < num_rows; i++) {
        if (csr.rowPtrs[i] - last > max_nz_items) max_nz_items = csr.rowPtrs[i] - last;
        nnzPerRow[i] = i < num_rows - 1 ? csr.rowPtrs[i + 1] - csr.rowPtrs[i] : num_cols - csr.rowPtrs[i];
    }

    // Pad rows to equal size

    unsigned int ell_cols[num_rows * max_nz_items] = {0};
    unsigned int ell_vals[num_rows * max_nz_items] = {0};


    // Store in column-major
    for (unsigned int i = 0; i < max_nz_items; i++) {
        for (unsigned int j = 0; j < num_rows; j++) {
            nnzPerRow[j] += 
            unsigned int col = i < nnzPerRow[j] ? csr.colIdx[j*num_rows + i] : 0;
            float val = i < nnzPerRow[j] ? csr.value[j*num_rows + i] : 0.0f;

            ell_cols[i*max_nz_items + j] = col;
            ell_vals[i*max_nz_items + j] = val;
        }
    }

    return {num_cols, nnzPerRow, ell_cols, ell_vals};

}

// TODO Convert ELL to ELL-COO hybrid __host__ convert_ell_hybrid() {}


__global__ void spmv_ell_kernel(ELLMatrix ellMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < ellMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < ellMatrix.nnzPerRow[row]; t++) {
            unsigned int i = t*ellMatrix.numRows + row;
            unsigned int col = ellMatrix.colIdx[col];
            float value = ellMatrix.value[i];
            sum += x[col]*value;
        }
        y[row] = sum;
    }
}