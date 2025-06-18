#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_LINE_LENGTH 1024
#define MAX_LINES 100000  // Reduced from 10000000 to prevent memory issues
#define MAX_META_LENGTH 32
#define MAX_META_LENGTH 32

__global__ void searchKernel(
    char *d_lines,
    char *d_pattern,
    int pattSize,
    int lineCount,
    int *d_matchIndices,
    int *d_matchCount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lineCount) return;

    char *line = d_lines + idx * MAX_LINE_LENGTH;

    for (int i = 0; i <= MAX_LINE_LENGTH - pattSize; i++) {
        int j;
        for (j = 0; j < pattSize; j++) {
            if (line[i + j] != d_pattern[j])
                break;
        }
        if (j == pattSize) {
            int pos = atomicAdd(d_matchCount, 1);
            d_matchIndices[pos] = idx;
            break;
        }
    }
}

int main() {
    char inputFileLocation[] = "/home/cj/HPC_data/Human_genome_preprocessed.fna";
    FILE *infile = NULL;
    char pattern[MAX_LINE_LENGTH];
    char buffer[MAX_LINE_LENGTH];
    
    // Declare all variables at the beginning to avoid goto bypassing initialization
    int lineCount = 0;
    char **lines = NULL;
    char (*chromoHost)[MAX_META_LENGTH] = NULL;
    char (*lineNumHost)[MAX_META_LENGTH] = NULL;
    char *h_allLines = NULL;
    int *h_matchIndices = NULL;
    int h_matchCount = 0;
    char *d_lines = NULL, *d_pattern = NULL;
    int *d_matchIndices = NULL, *d_matchCount = NULL;
    cudaError_t cudaStatus = cudaSuccess;
    int blockSize = 256;
    int numBlocks = 0;
    float milliseconds = 0;
    cudaEvent_t start, stop;
    
    // Use dynamic allocation for arrays
    lines = (char **)malloc(MAX_LINES * sizeof(char *));
    chromoHost = (char (*)[MAX_META_LENGTH])malloc(MAX_LINES * sizeof(char[MAX_META_LENGTH]));
    lineNumHost = (char (*)[MAX_META_LENGTH])malloc(MAX_LINES * sizeof(char[MAX_META_LENGTH]));
    
    if (!lines || !chromoHost || !lineNumHost) {
        fprintf(stderr, "Failed to allocate memory for arrays\n");
        goto Error;
    }
    
    // Initialize pointers to NULL
    memset(lines, 0, MAX_LINES * sizeof(char *));

    printf("Enter pattern to search: ");
    scanf("%s", pattern);

    infile = fopen(inputFileLocation, "r");
    if (!infile) {
        perror("Error opening file");
        free(lines);
        free(chromoHost);
        free(lineNumHost);
        return 1;
    }

    while (fgets(buffer, sizeof(buffer), infile) && lineCount < MAX_LINES) {
        lines[lineCount] = strdup(buffer);
        if (!lines[lineCount]) {
            fprintf(stderr, "Memory allocation failed for line %d\n", lineCount);
            break;
        }

        // Extract metadata
        char *underscore = strchr(buffer, '_');
        char *pipe = strchr(buffer, '|');
        if (underscore && pipe && underscore < pipe) {
            int len_chromo = underscore - buffer;
            strncpy(chromoHost[lineCount], buffer, len_chromo);
            chromoHost[lineCount][len_chromo] = '\0';

            int len_lineNum = pipe - underscore - 1;
            strncpy(lineNumHost[lineCount], underscore + 1, len_lineNum);
            lineNumHost[lineCount][len_lineNum] = '\0';
        } else {
            strcpy(chromoHost[lineCount], "NA");
            strcpy(lineNumHost[lineCount], "NA");
        }

        lineCount++;
    }
    
    if (infile) {
        fclose(infile);
        infile = NULL;
    }

    printf("Read %d lines from file\n", lineCount);
    
    if (lineCount == 0) {
        fprintf(stderr, "No lines read from file\n");
        for (int i = 0; i < lineCount; i++) free(lines[i]);
        free(lines);
        free(chromoHost);
        free(lineNumHost);
        return 1;
    }

    // Flattened data for device
    h_allLines = (char *)malloc(lineCount * MAX_LINE_LENGTH);
    if (!h_allLines) {
        fprintf(stderr, "Failed to allocate memory for flattened lines\n");
        for (int i = 0; i < lineCount; i++) free(lines[i]);
        free(lines);
        free(chromoHost);
        free(lineNumHost);
        return 1;
    }
    
    for (int i = 0; i < lineCount; i++) {
        strncpy(h_allLines + i * MAX_LINE_LENGTH, lines[i], MAX_LINE_LENGTH - 1);
        h_allLines[i * MAX_LINE_LENGTH + MAX_LINE_LENGTH - 1] = '\0'; // Ensure null termination
    }

    // Allocate device memory with error checking
    cudaStatus = cudaMalloc(&d_lines, lineCount * MAX_LINE_LENGTH);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_lines failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaMalloc(&d_pattern, strlen(pattern) + 1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_pattern failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaMalloc(&d_matchIndices, lineCount * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_matchIndices failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaMalloc(&d_matchCount, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_matchCount failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy data to device with error checking
    cudaStatus = cudaMemcpy(d_lines, h_allLines, lineCount * MAX_LINE_LENGTH, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to d_lines failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaMemcpy(d_pattern, pattern, strlen(pattern) + 1, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to d_pattern failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaMemset(d_matchCount, 0, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset for d_matchCount failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Launch kernel
    numBlocks = (lineCount + blockSize - 1) / blockSize;

    printf("Searching for pattern: %s\n", pattern);
    printf("============================================================================\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    searchKernel<<<numBlocks, blockSize>>>(d_lines, d_pattern, (int)strlen(pattern), lineCount, d_matchIndices, d_matchCount);
    
    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results back
    cudaStatus = cudaMemcpy(&h_matchCount, d_matchCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy from d_matchCount failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Allocate memory for match indices
    if (h_matchCount > 0) {
        h_matchIndices = (int *)malloc(h_matchCount * sizeof(int));
        if (!h_matchIndices) {
            fprintf(stderr, "Failed to allocate memory for match indices\n");
            goto Error;
        }
        
        cudaStatus = cudaMemcpy(h_matchIndices, d_matchIndices, h_matchCount * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy from d_matchIndices failed: %s\n", cudaGetErrorString(cudaStatus));
            free(h_matchIndices);
            h_matchIndices = NULL;
            goto Error;
        }
    }

    // Print matches
    for (int i = 0; i < h_matchCount; i++) {
        int idx = h_matchIndices[i];
        if (idx >= 0 && idx < lineCount) {
            printf("Pattern found at chromosome %s, at line %s\n", chromoHost[idx], lineNumHost[idx]);
        }
    }

    printf("============================================================================\n");
    printf("Total matches found: %d\n", h_matchCount);
    printf("Time taken for CUDA search: %.6f seconds\n", milliseconds / 1000.0);

    // Cleanup
    if (h_matchIndices) free(h_matchIndices);
    
Error:
    // Free all allocated memory
    if (d_lines) cudaFree(d_lines);
    if (d_pattern) cudaFree(d_pattern);
    if (d_matchIndices) cudaFree(d_matchIndices);
    if (d_matchCount) cudaFree(d_matchCount);
    
    if (h_allLines) free(h_allLines);
    
    if (lines) {
        for (int i = 0; i < lineCount; i++) {
            if (lines[i]) free(lines[i]);
        }
        free(lines);
    }
    
    if (chromoHost) free(chromoHost);
    if (lineNumHost) free(lineNumHost);
    
    if (infile) fclose(infile);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Reset device to clear any errors
    cudaDeviceReset();

    return (cudaStatus == cudaSuccess) ? 0 : 1;
}
