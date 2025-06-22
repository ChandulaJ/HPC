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
    int totalLineCount = 0;        // Total lines processed
    int totalMatchCount = 0;       // Total matches found across all batches
    int currentBatch = 0;          // Current batch number
    char (*chromoHost)[MAX_META_LENGTH] = NULL;  // Metadata for current batch
    char (*lineNumHost)[MAX_META_LENGTH] = NULL; // Line numbers for current batch
    char *h_batchLines = NULL;     // Host memory for batch (pinned for faster transfers)
    int *h_matchIndices = NULL;    // Match indices from current batch
    int *h_matchOffsets = NULL;    // Offsets for global line numbers
    int h_matchCount = 0;          // Match count from current batch
    char *d_lines = NULL;          // Device memory for lines
    char *d_pattern = NULL;        // Device memory for pattern
    int *d_matchIndices = NULL;    // Device memory for match indices
    int *d_matchCount = NULL;      // Device memory for match count
    cudaStream_t streams[NUM_STREAMS]; // CUDA streams for overlapping operations
    cudaError_t cudaStatus = cudaSuccess;
    int numBlocks = 0;
    float totalMilliseconds = 0;
    cudaEvent_t start, stop;
    
    // Optimize CUDA device settings for RTX 3050 Laptop GPU
    // These settings prioritize shared memory and enable asynchronous operations
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleYield | cudaDeviceLmemResizeToMax);
    
    // Create CUDA streams for overlapping operations
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (cudaStreamCreate(&streams[i]) != cudaSuccess) {
            fprintf(stderr, "Failed to create CUDA stream %d\n", i);
            goto Error;
        }
    }
    
    // Create CUDA events for timing
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA events\n");
        goto Error;
    }
    
    // Get and display GPU information
    cudaDeviceProp prop;
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
    
    // Get search pattern from user
    printf("Enter pattern to search: ");
    if (scanf("%s", pattern) != 1) {
        fprintf(stderr, "Error reading pattern\n");
        goto Error;
    }
    
    printf("Searching for pattern: %s\n", pattern);
    printf("Pattern size: %d\n", (int)strlen(pattern));
    printf("============================================================================\n");
    
    // Allocate and initialize device memory
    cudaStatus = cudaMalloc(&d_pattern, strlen(pattern) + 1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_pattern failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaMemcpy(d_pattern, pattern, strlen(pattern) + 1, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to d_pattern failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaMalloc(&d_matchCount, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_matchCount failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Allocate pinned memory for metadata (improves transfer performance)
    cudaStatus = cudaMallocHost((void**)&chromoHost, BATCH_SIZE * sizeof(char[MAX_META_LENGTH]));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost for chromoHost failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaMallocHost((void**)&lineNumHost, BATCH_SIZE * sizeof(char[MAX_META_LENGTH]));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost for lineNumHost failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaMallocHost((void**)&h_matchOffsets, BATCH_SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost for h_matchOffsets failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Use pinned memory for batch lines (critical for fast transfers to GPU)
    cudaStatus = cudaMallocHost(&h_batchLines, BATCH_SIZE * MAX_LINE_LENGTH);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost for h_batchLines failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Allocate device memory for lines and match indices
    cudaStatus = cudaMalloc(&d_lines, BATCH_SIZE * MAX_LINE_LENGTH);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_lines failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaMalloc(&d_matchIndices, BATCH_SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_matchIndices failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Allocate read buffers for file I/O
    buffer = (char*)malloc(READ_BUFFER_SIZE);
    lineBuffer = (char*)malloc(MAX_LINE_LENGTH);
    if (!buffer || !lineBuffer) {
        fprintf(stderr, "Failed to allocate read buffers\n");
        goto Error;
    }
    
    // Open input file
    infile = fopen(inputFileLocation, "r");
    if (!infile) {
        perror("Error opening file");
        goto Error;
    }
    
    // Set a larger buffer for file I/O
    if (setvbuf(infile, buffer, _IOFBF, READ_BUFFER_SIZE) != 0) {
        fprintf(stderr, "Warning: Could not set file buffer, using default buffering\n");
    }
    
    // Start global timer
    cudaEventRecord(start, 0);
    
    printf("============================================================================\n");
    
    // Process file in batches
    while (!feof(infile)) {
        int batchLineCount = 0;
        
        // Read a batch of lines with optimized I/O
        while (batchLineCount < BATCH_SIZE && fgets(lineBuffer, MAX_LINE_LENGTH, infile)) {
            // Copy line immediately to h_batchLines to avoid extra memory allocation and copies
            int lineLen = strlen(lineBuffer);
            memcpy(h_batchLines + batchLineCount * MAX_LINE_LENGTH, lineBuffer, lineLen + 1); // +1 for null terminator
            
            // Extract metadata directly from lineBuffer to avoid extra copies
            char *underscore = strchr(lineBuffer, '_');
            char *pipe = strchr(lineBuffer, '|');
            if (underscore && pipe && underscore < pipe) {
                int len_chromo = underscore - lineBuffer;
                strncpy(chromoHost[batchLineCount], lineBuffer, len_chromo);
                chromoHost[batchLineCount][len_chromo] = '\0';
                
                int len_lineNum = pipe - underscore - 1;
                strncpy(lineNumHost[batchLineCount], underscore + 1, len_lineNum);
                lineNumHost[batchLineCount][len_lineNum] = '\0';
            } else {
                strcpy(chromoHost[batchLineCount], "NA");
                strcpy(lineNumHost[batchLineCount], "NA");
            }
            
            // Save line offset for global indexing
            h_matchOffsets[batchLineCount] = totalLineCount + batchLineCount;
            
            batchLineCount++;
        }
        
        if (batchLineCount == 0) {
            // No more lines to read
            break;
        }
        
        // Just ensure all buffers have proper null termination
        for (int i = 0; i < batchLineCount; i++) {
            h_batchLines[i * MAX_LINE_LENGTH + MAX_LINE_LENGTH - 1] = '\0'; // Ensure null termination
        }
        
        // Use stream for this batch cycle
        cudaStream_t currentStream = streams[currentBatch % NUM_STREAMS];
        
        // Copy batch data to device using stream
        cudaStatus = cudaMemcpyAsync(d_lines, h_batchLines, batchLineCount * MAX_LINE_LENGTH, 
                                   cudaMemcpyHostToDevice, currentStream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync to d_lines failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        


        // Reset match count for this batch
        cudaStatus = cudaMemsetAsync(d_matchCount, 0, sizeof(int), currentStream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemsetAsync for d_matchCount failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        // Launch kernel with settings optimized for RTX 3050 Laptop GPU
        // Calculate optimal block count based on GPU's multiprocessor count
        numBlocks = (batchLineCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        // Ensure we have enough blocks to keep all SMs busy (RTX 3050 has ~16 SMs)
        numBlocks = min(numBlocks, (batchLineCount + 31) / 32);
        
        int sharedMemSize = SHARED_MEM_SIZE; // Only need space for the pattern
        searchKernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize, currentStream>>>(
            d_lines, d_pattern, (int)strlen(pattern), 
            batchLineCount, d_matchIndices, d_matchCount);
        
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
        

        
        // Copy match count result back from device asynchronously
        cudaStatus = cudaMemcpyAsync(&h_matchCount, d_matchCount, sizeof(int), 
                                   cudaMemcpyDeviceToHost, currentStream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync from d_matchCount failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        // Need to synchronize on the stream before using h_matchCount
        cudaStreamSynchronize(currentStream);
        

        
        if (h_matchCount > 0) {
            // Allocate memory for match indices
            h_matchIndices = (int *)malloc(h_matchCount * sizeof(int));
            if (!h_matchIndices) {
                fprintf(stderr, "Failed to allocate memory for match indices\n");
                goto Error;
            }
            
            // Copy match indices from device asynchronously
            cudaStatus = cudaMemcpyAsync(h_matchIndices, d_matchIndices, h_matchCount * sizeof(int), 
                                       cudaMemcpyDeviceToHost, currentStream);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpyAsync from d_matchIndices failed: %s\n", cudaGetErrorString(cudaStatus));
                free(h_matchIndices);
                h_matchIndices = NULL;
                goto Error;
            }
            
            // Wait for transfer to complete
            cudaStreamSynchronize(currentStream);
            
            // Print matches from this batch
            for (int i = 0; i < h_matchCount; i++) {
                int idx = h_matchIndices[i];
                if (idx >= 0 && idx < batchLineCount) {
                    printf("Pattern found at chromosome %s, at line %s\n", 
                           chromoHost[idx], lineNumHost[idx]);
                }
            }
            
            // Free match indices for this batch
            free(h_matchIndices);
            h_matchIndices = NULL;
            
            // Update total match count
            totalMatchCount += h_matchCount;
        }
        
        // Update total line count
        totalLineCount += batchLineCount;
        currentBatch++;
    
        fflush(stdout);
    }
    
    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&totalMilliseconds, start, stop);
    
    if (infile) {
        fclose(infile);
        infile = NULL;
    }
    
    printf("\nFinished processing %d lines in %d batches\n", totalLineCount, currentBatch);
    printf("============================================================================\n");
    printf("Total matches found: %d\n", totalMatchCount);
    printf("Time taken for CUDA search: %.6f seconds\n", totalMilliseconds / 1000.0);
    
Error:
    // Free all allocated memory
    if (d_lines) cudaFree(d_lines);
    if (d_pattern) cudaFree(d_pattern);
    if (d_matchIndices) cudaFree(d_matchIndices);
    if (d_matchCount) cudaFree(d_matchCount);
    
    if (h_batchLines) cudaFreeHost(h_batchLines);
    if (h_matchIndices) free(h_matchIndices);
    if (h_matchOffsets) cudaFreeHost(h_matchOffsets);
    if (chromoHost) cudaFreeHost(chromoHost);
    if (lineNumHost) cudaFreeHost(lineNumHost);
    
    // Free read buffers
    if (buffer) free(buffer);
    if (lineBuffer) free(lineBuffer);
    
    if (infile) fclose(infile);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Clean up streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    // Reset device to clear any errors
    cudaDeviceReset();

    return (cudaStatus == cudaSuccess) ? 0 : 1;
}
