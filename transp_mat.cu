#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*
 * See section "B. 19 Launch Bounds" from "CUDA C Programming Guide" for more
 * information about the optimal launch bounds, which differ across the major
 * architecture revisions
 */
#define THREADS_PER_BLOCK_2D 16

/* Simple utility function to check for CUDA runtime errors */
void checkCUDAError(const char* msg);

/* Host function that transposes a matrix */
void transpose_cpu(const char* mat_in, char* mat_out, unsigned int rows,
        unsigned int cols);

/* Kernel code */
__global__ void transpose_gpu(const char* mat_in, char* mat_out,
        unsigned int rows, unsigned int cols) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;

        mat_out[trans_pos] = mat_in[pos];
    }
}

int main(int argc, char** argv) {

    /* Process command-line arguments */
    if (argc != 3) {
        fprintf(stderr, "Usage: %s rows columns\n", argv[0]);
        fprintf(stderr, "       rows is the number of rows of the input matrix\n");
        fprintf(stderr, "       columns is the number of columns of the input matrix\n");
        return EXIT_FAILURE;
    }

    cudaEvent_t start, stop;
    float elapsed_time_ms;

    unsigned int rows = atoi(argv[1]);
    unsigned int cols = atoi(argv[2]);

    /* Pointer for host memory */
    char *h_mat_in, *h_mat_out;
    size_t mat_size = rows * cols * sizeof(char);

    /* Pointer for device memory */
    char *dev_mat_in, *dev_mat_out;

    /* Allocate host and device memory */
    h_mat_in = (char *) malloc(mat_size);
    h_mat_out = (char *) malloc(mat_size);

    cudaMalloc(&dev_mat_in, mat_size);
    cudaMalloc(&dev_mat_out, mat_size);

    /* Check for any CUDA errors */
    checkCUDAError("cudaMalloc");

    /* Fixed seed for illustration */
    srand(2047);

    /* Initialize host memory */
    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            h_mat_in[i * cols + j] =  rand() % (rows * cols);
            //printf("%d\t", h_mat_in[i * cols + j]);
        }
        //printf("\n");
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*------------------------ COMPUTATION ON CPU ----------------------------*/

    cudaEventRecord(start, 0);
    // cudaEventSynchronize(start); needed?

    transpose_cpu(h_mat_in, h_mat_out, rows, cols);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    /* Verify the correctness of transposed matrix computed on GPU */
    for (unsigned int i = 0; i < cols; ++i) {
        for (unsigned int j = 0; j < rows; ++j) {
            assert(h_mat_out[i * rows + j] == h_mat_in[j * cols + i]);
        }
    }

    printf("Time to transpose a matrix of %dx%d on CPU: %f ms.\n\n", rows, cols,
                elapsed_time_ms);

    /*------------------------ COMPUTATION ON GPU ----------------------------*/

    /* Host to device memory copy */
    cudaMemcpy(dev_mat_in, h_mat_in, mat_size, cudaMemcpyHostToDevice);

    /* Check for any CUDA errors */
    checkCUDAError("cudaMemcpy");

    /* Set grid and block dimensions properly */
    unsigned int grid_rows = (rows + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D;
    unsigned int grid_cols = (cols + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);

    cudaEventRecord(start, 0);
    // cudaEventSynchronize(start); needed?

    /* Launch kernel */
    transpose_gpu<<<dimGrid, dimBlock>>>(dev_mat_in, dev_mat_out, rows, cols);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    /* Check for any CUDA errors */
    checkCUDAError("kernel invocation");

    /* device to host copy */
    cudaMemcpy(h_mat_out, dev_mat_out, mat_size, cudaMemcpyDeviceToHost);

    /* Check for any CUDA errors */
    checkCUDAError("cudaMemcpy");

    /* Verify the correctness of transposed matrix computed on GPU */
    for (unsigned int i = 0; i < cols; ++i) {
        for (unsigned int j = 0; j < rows; ++j) {
            assert(h_mat_out[i * rows + j] == h_mat_in[j * cols + i]);
        }
    }

    printf("Time to transpose a matrix of %dx%d on GPU: %f ms.\n\n", rows, cols,
            elapsed_time_ms);

    /* Output the transposed matrix
    for (unsigned int i = 0; i < cols; ++i) {
        for (unsigned int j = 0; j < rows; ++j) {
            printf("%d\t", (int) h_mat_out[i * rows + j]);
        }
        printf("\n");
    }*/

    /* Free host and device memory */
    free(h_mat_in);
    free(h_mat_out);
    cudaFree(dev_mat_in);
    cudaFree(dev_mat_out);

    /* Check for any CUDA errors */
    checkCUDAError("cudaFree");
}

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void transpose_cpu(const char* mat_in, char* mat_out, unsigned int rows,
        unsigned int cols) {
    for (unsigned int i = 0; i < cols; ++i) {
        for (unsigned int j = 0; j < rows; ++j) {
            mat_out[i * rows + j] = mat_in[j * cols + i];
        }
    }
}
