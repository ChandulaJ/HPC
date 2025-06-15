#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_LINE_LENGTH 1024
#define MAX_LINES 10000000
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
    FILE *infile;
    char pattern[MAX_LINE_LENGTH];
    char buffer[MAX_LINE_LENGTH];
    char *lines[MAX_LINES];
    char chromoHost[MAX_LINES][MAX_META_LENGTH];
    char lineNumHost[MAX_LINES][MAX_META_LENGTH];
    int lineCount = 0;

    printf("Enter pattern to search: ");
    scanf("%s", pattern);

    infile = fopen(inputFileLocation, "r");
    if (!infile) {
        perror("Error opening file");
        return 1;
    }

    while (fgets(buffer, sizeof(buffer), infile) && lineCount < MAX_LINES) {
        lines[lineCount] = strdup(buffer);

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
    fclose(infile);

    // Flattened data for device
    char *h_allLines = (char *)malloc(lineCount * MAX_LINE_LENGTH);
    for (int i = 0; i < lineCount; i++) {
        strncpy(h_allLines + i * MAX_LINE_LENGTH, lines[i], MAX_LINE_LENGTH);
    }

    // Allocate device memory
    char *d_lines, *d_pattern;
    int *d_matchIndices, *d_matchCount;
    cudaMalloc(&d_lines, lineCount * MAX_LINE_LENGTH);
    cudaMalloc(&d_pattern, strlen(pattern));
    cudaMalloc(&d_matchIndices, lineCount * sizeof(int));
    cudaMalloc(&d_matchCount, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_lines, h_allLines, lineCount * MAX_LINE_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern, strlen(pattern), cudaMemcpyHostToDevice);
    cudaMemset(d_matchCount, 0, sizeof(int));

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (lineCount + blockSize - 1) / blockSize;

    printf("Searching for pattern: %s\n", pattern);
    printf("============================================================================\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    searchKernel<<<numBlocks, blockSize>>>(d_lines, d_pattern, strlen(pattern), lineCount, d_matchIndices, d_matchCount);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results back
    int h_matchCount;
    cudaMemcpy(&h_matchCount, d_matchCount, sizeof(int), cudaMemcpyDeviceToHost);
    int *h_matchIndices = (int *)malloc(h_matchCount * sizeof(int));
    cudaMemcpy(h_matchIndices, d_matchIndices, h_matchCount * sizeof(int), cudaMemcpyDeviceToHost);

    // Print matches
    for (int i = 0; i < h_matchCount; i++) {
        int idx = h_matchIndices[i];
        printf("Pattern found at chromosome %s, at line %s\n", chromoHost[idx], lineNumHost[idx]);
    }

    printf("============================================================================\n");
    printf("Total matches found: %d\n", h_matchCount);
    printf("Time taken for CUDA search: %.6f seconds\n", milliseconds / 1000.0);

    // Cleanup
    for (int i = 0; i < lineCount; i++) free(lines[i]);
    free(h_allLines);
    free(h_matchIndices);
    cudaFree(d_lines);
    cudaFree(d_pattern);
    cudaFree(d_matchIndices);
    cudaFree(d_matchCount);

    return 0;
}
