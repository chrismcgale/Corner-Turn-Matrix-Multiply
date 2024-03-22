#define TILE_WIDTH  32


// NOTE: key difference here is that N has been stored in a column-major layout
// Assumes square matrices
__global__ void matrixMultiplyKernel(float* M, float* N, float* P, int width) {

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;

    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;

    // Loop over the tiles from M and N required to compute P
    float pValue = 0.0f;
    for(int ph = 0; ph < ceil(width/(float)TILE_WIDTH); ph++) {

        // Collaboratively load Mds and Nds
        if ((row < width) && (ph*TILE_WIDTH + tx < width)) {
            Mds[ty][tx] = M[row*width + ph*TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }
        
        if ((col < width) && (ph*TILE_WIDTH + ty < width)) {
            // Index for N used to be '(ph*TILE_WIDTH + ty)*width + col' changed to improve memory coalescing due to column-major layout, 
            Nds[ty][tx] = N[col*width + ph*TILE_WIDTH + ty];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __synchthreads();

        // Once stored in shared memory, coalescing is no longer an issue (due to size, locality, etc.)
        for (int k = 0; k < TILE_WIDTH; k++) {
            pValue += Mds[ty][k] * Nds[k][tx];
        }
        __synchThreads();
    }
    P[row*width + col] = pValue;
}