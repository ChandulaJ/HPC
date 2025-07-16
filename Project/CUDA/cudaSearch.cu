#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_LINE_LENGTH 105  // Optimized for typical genomic data line lengths and better memory alignment
#define BATCH_SIZE 65536     // Optimized for RTX 3050 Laptop (4GB VRAM, ~2048 CUDA cores)
#define MAX_META_LENGTH 32
#define READ_BUFFER_SIZE (16 * 1024 * 1024)  // 16MB read buffer for better I/O performance

#define THREADS_PER_BLOCK 512 // RTX 3050 supports up to 1024 threads per block (Ampere arch)
#define SHARED_MEM_SIZE 256   // Size of shared memory for pattern (naive algorithm)
#define NUM_STREAMS 4         // Multiple streams for better overlapping on RTX 3050

// Highly optimized naive string matching implementation for CUDA
__global__ void searchKernel(
    const char *d_lines,
    const char *d_pattern,
    int pattSize,
    int lineCount,
    int *d_matchIndices,
    int *d_matchCount
) {
    __shared__ char sharedPattern[SHARED_MEM_SIZE];  // Shared memory for pattern
    
    // Cooperative loading of pattern into shared memory (coalesced)
    // Each warp loads a section of the pattern - reduces bank conflicts
    if (threadIdx.x < pattSize) {
        sharedPattern[threadIdx.x] = d_pattern[threadIdx.x];
    }
    
    __syncthreads(); // Wait for all threads to finish loading pattern
    
    // Each thread processes one line
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lineCount) return;
    
    const char *line = d_lines + idx * MAX_LINE_LENGTH;
    
    // Simple naive string matching algorithm
    // For each possible starting position in the line
    for (int i = 0; i <= MAX_LINE_LENGTH - pattSize; i++) {
        // Check if pattern matches at this position
        bool matched = true;
        
        for (int j = 0; j < pattSize; j++) {
            if (line[i + j] != sharedPattern[j]) {
                matched = false;
                break;
            }
        }
        
        if (matched) {
            // Found a match
            int pos = atomicAdd(d_matchCount, 1);
            if (pos < BATCH_SIZE) { // Prevent array overruns
                d_matchIndices[pos] = idx;
            }
            break; // Only record the first match in each line
        }
    }
}

int main() {
    char inputFileLocation[] = "/home/cj/HPC_data/Human_genome_preprocessed.fna";
    FILE *infile = NULL;
    char pattern[MAX_LINE_LENGTH];
    char *buffer = NULL;        // Large read buffer for file I/O
    char *lineBuffer = NULL;    // Temporary buffer for line processing
    
    // Declare all variables at the beginning
    int totalLineCount = 0, totalMatchCount = 0, currentBatch = 0, h_matchCount = 0, numBlocks = 0;
    float totalMilliseconds = 0;
    char (*chromoHost)[MAX_META_LENGTH] = NULL, (*lineNumHost)[MAX_META_LENGTH] = NULL;
    char *h_batchLines = NULL, *d_lines = NULL, *d_pattern = NULL;
    int *h_matchIndices = NULL, *d_matchIndices = NULL, *d_matchCount = NULL;
    cudaStream_t streams[NUM_STREAMS];
    cudaError_t cudaStatus = cudaSuccess;
    cudaEvent_t start, stop;
    
    // Initialize CUDA settings and create streams/events
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleYield | cudaDeviceLmemResizeToMax);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (cudaStreamCreate(&streams[i]) != cudaSuccess) {
            fprintf(stderr, "Failed to create CUDA stream %d\n", i);
            goto Error;
        }
    }
    
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA events\n");
        goto Error;
    }
    
    // Get GPU properties and optimize kernel configuration
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        cudaFuncSetAttribute(searchKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 
                           cudaSharedmemCarveoutMaxShared);
    } else {
        fprintf(stderr, "Warning: Could not get device properties\n");
    }
    
        // Get and display GPU information
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        printf("Using GPU: %s\n", prop.name);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Shared Memory per Block: %lu KB\n", prop.sharedMemPerBlock / 1024);
        
        // Optimize kernel configuration based on GPU properties
        cudaFuncSetAttribute(searchKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 
                           cudaSharedmemCarveoutMaxShared);
    } else {
        fprintf(stderr, "Warning: Could not get device properties\n");
    }

    // Get search pattern and allocate device memory
    printf("Enter pattern to search: ");
    if (scanf("%s", pattern) != 1) {
        fprintf(stderr, "Error reading pattern\n");
        goto Error;
    }
    printf("Searching for pattern: %s\n", pattern);
    printf("============================================================================\n");
    
    if ((cudaStatus = cudaMalloc(&d_pattern, strlen(pattern) + 1)) != cudaSuccess ||
        (cudaStatus = cudaMemcpy(d_pattern, pattern, strlen(pattern) + 1, cudaMemcpyHostToDevice)) != cudaSuccess ||
        (cudaStatus = cudaMalloc(&d_matchCount, sizeof(int))) != cudaSuccess) {
        fprintf(stderr, "Device memory allocation/copy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Allocate pinned host memory and device memory
    if ((cudaStatus = cudaMallocHost((void**)&chromoHost, BATCH_SIZE * sizeof(char[MAX_META_LENGTH]))) != cudaSuccess ||
        (cudaStatus = cudaMallocHost((void**)&lineNumHost, BATCH_SIZE * sizeof(char[MAX_META_LENGTH]))) != cudaSuccess ||
        (cudaStatus = cudaMallocHost(&h_batchLines, BATCH_SIZE * MAX_LINE_LENGTH)) != cudaSuccess ||
        (cudaStatus = cudaMalloc(&d_lines, BATCH_SIZE * MAX_LINE_LENGTH)) != cudaSuccess ||
        (cudaStatus = cudaMalloc(&d_matchIndices, BATCH_SIZE * sizeof(int))) != cudaSuccess) {
        fprintf(stderr, "Memory allocation failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Allocate read buffers and open input file
    buffer = (char*)malloc(READ_BUFFER_SIZE);
    lineBuffer = (char*)malloc(MAX_LINE_LENGTH);
    if (!buffer || !lineBuffer) {
        fprintf(stderr, "Failed to allocate read buffers\n");
        goto Error;
    }
    
    if (!(infile = fopen(inputFileLocation, "r"))) {
        perror("Error opening file");
        goto Error;
    }
    
    if (setvbuf(infile, buffer, _IOFBF, READ_BUFFER_SIZE) != 0) {
        fprintf(stderr, "Warning: Could not set file buffer, using default buffering\n");
    }
    cudaEventRecord(start, 0);
    
    // Process file in batches
    while (!feof(infile)) {
        int batchLineCount = 0;
        
        // Read a batch of lines with optimized I/O
        while (batchLineCount < BATCH_SIZE && fgets(lineBuffer, MAX_LINE_LENGTH, infile)) {
            // Copy line and extract metadata
            int lineLen = strlen(lineBuffer);
            memcpy(h_batchLines + batchLineCount * MAX_LINE_LENGTH, lineBuffer, lineLen + 1);
            
            char *underscore = strchr(lineBuffer, '_'), *pipe = strchr(lineBuffer, '|');
            if (underscore && pipe && underscore < pipe) {
                int len_chromo = underscore - lineBuffer, len_lineNum = pipe - underscore - 1;
                strncpy(chromoHost[batchLineCount], lineBuffer, len_chromo);
                chromoHost[batchLineCount][len_chromo] = '\0';
                strncpy(lineNumHost[batchLineCount], underscore + 1, len_lineNum);
                lineNumHost[batchLineCount][len_lineNum] = '\0';
            } else {
                strcpy(chromoHost[batchLineCount], "NA");
                strcpy(lineNumHost[batchLineCount], "NA");
            }
            batchLineCount++;
        }
        
        if (batchLineCount == 0) break;
        
        cudaStream_t currentStream = streams[currentBatch % NUM_STREAMS];
        
        // Copy data to device and launch kernel
        if ((cudaStatus = cudaMemcpyAsync(d_lines, h_batchLines, batchLineCount * MAX_LINE_LENGTH, 
                                        cudaMemcpyHostToDevice, currentStream)) != cudaSuccess ||
            (cudaStatus = cudaMemsetAsync(d_matchCount, 0, sizeof(int), currentStream)) != cudaSuccess) {
            fprintf(stderr, "Device memory operation failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        // Launch kernel
        numBlocks = (batchLineCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        numBlocks = min(numBlocks, (batchLineCount + 31) / 32);
        
        searchKernel<<<numBlocks, THREADS_PER_BLOCK, SHARED_MEM_SIZE, currentStream>>>(
            d_lines, d_pattern, (int)strlen(pattern), batchLineCount, d_matchIndices, d_matchCount);
        
        if ((cudaStatus = cudaGetLastError()) != cudaSuccess ||
            (cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) {
            fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        // Copy results back and process matches
        if ((cudaStatus = cudaMemcpyAsync(&h_matchCount, d_matchCount, sizeof(int), 
                                        cudaMemcpyDeviceToHost, currentStream)) != cudaSuccess) {
            fprintf(stderr, "Match count copy failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        cudaStreamSynchronize(currentStream);
        
        if (h_matchCount > 0) {
            if (!(h_matchIndices = (int *)malloc(h_matchCount * sizeof(int)))) {
                fprintf(stderr, "Failed to allocate memory for match indices\n");
                goto Error;
            }
            
            if ((cudaStatus = cudaMemcpyAsync(h_matchIndices, d_matchIndices, h_matchCount * sizeof(int), 
                                            cudaMemcpyDeviceToHost, currentStream)) != cudaSuccess) {
                fprintf(stderr, "Match indices copy failed: %s\n", cudaGetErrorString(cudaStatus));
                free(h_matchIndices);
                h_matchIndices = NULL;
                goto Error;
            }
            cudaStreamSynchronize(currentStream);
            
            for (int i = 0; i < h_matchCount; i++) {
                int idx = h_matchIndices[i];
                if (idx >= 0 && idx < batchLineCount) {
                    printf("Pattern found at chromosome %s, at line %s\n", 
                           chromoHost[idx], lineNumHost[idx]);
                }
            }
            free(h_matchIndices);
            h_matchIndices = NULL;
            totalMatchCount += h_matchCount;
        }
        totalLineCount += batchLineCount;
        currentBatch++;
        fflush(stdout);
    }
    
    // Cleanup and display results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&totalMilliseconds, start, stop);
    
    if (infile) { fclose(infile); infile = NULL; }
    
    printf("\nFinished processing %d lines in %d batches\n", totalLineCount, currentBatch);
    printf("============================================================================\n");
    printf("Total matches found: %d\n", totalMatchCount);
    printf("Time taken for CUDA search: %.6f seconds\n", totalMilliseconds / 1000.0);
    
Error:
    // Cleanup
    if (d_lines) cudaFree(d_lines);
    if (d_pattern) cudaFree(d_pattern);
    if (d_matchIndices) cudaFree(d_matchIndices);
    if (d_matchCount) cudaFree(d_matchCount);
    if (h_batchLines) cudaFreeHost(h_batchLines);
    if (h_matchIndices) free(h_matchIndices);
    if (chromoHost) cudaFreeHost(chromoHost);
    if (lineNumHost) cudaFreeHost(lineNumHost);
    if (buffer) free(buffer);
    if (lineBuffer) free(lineBuffer);
    if (infile) fclose(infile);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamDestroy(streams[i]);
    cudaDeviceReset();

    return (cudaStatus == cudaSuccess) ? 0 : 1;
}
