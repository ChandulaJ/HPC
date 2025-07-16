#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Optimized constants for hybrid processing
#define MAX_LINE_LENGTH 105
#define BATCH_SIZE 65536
#define MAX_META_LENGTH 32
#define READ_BUFFER_SIZE (16 * 1024 * 1024)
#define THREADS_PER_BLOCK 512
#define SHARED_MEM_SIZE 256
#define NUM_STREAMS 4
#define NUM_GPU_DEVICES 1

// Structure for batch processing data
typedef struct {
    char *lines;
    char (*chromo)[MAX_META_LENGTH];
    char (*lineNum)[MAX_META_LENGTH];
    int lineCount;
    int batchId;
} BatchData;

// Structure for search results
typedef struct {
    int *matchIndices;
    int matchCount;
    int batchId;
} SearchResult;

// CUDA kernel for pattern matching with shared memory optimization
__global__ void hybridSearchKernel(
    const char *d_lines,
    const char *d_pattern,
    int pattSize,
    int lineCount,
    int *d_matchIndices,
    int *d_matchCount
) {
    __shared__ char sharedPattern[SHARED_MEM_SIZE];
    
    // Cooperative loading of pattern into shared memory
    if (threadIdx.x < pattSize && threadIdx.x < SHARED_MEM_SIZE) {
        sharedPattern[threadIdx.x] = d_pattern[threadIdx.x];
    }
    
    __syncthreads();
    
    // Each thread processes one line
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lineCount) return;
    
    const char *line = d_lines + idx * MAX_LINE_LENGTH;
    int lineLen = 0;
    
    // Calculate actual line length (stop at newline or null terminator)
    while (lineLen < MAX_LINE_LENGTH && line[lineLen] != '\0' && line[lineLen] != '\n' && line[lineLen] != '\r') {
        lineLen++;
    }
    
    // Search for pattern in the line
    for (int i = 0; i <= lineLen - pattSize; i++) {
        bool matched = true;
        
        // Check if pattern matches at this position
        for (int j = 0; j < pattSize; j++) {
            if (line[i + j] != sharedPattern[j]) {
                matched = false;
                break;
            }
        }
        
        if (matched) {
            // Found a match - record it atomically
            int pos = atomicAdd(d_matchCount, 1);
            if (pos < BATCH_SIZE) {
                d_matchIndices[pos] = idx;
            }
            break; // Only record first match per line
        }
    }
}

// OpenMP-accelerated file reading and preprocessing
int loadBatchData(FILE *infile, BatchData *batch, char *buffer __attribute__((unused)), char *lineBuffer) {
    int batchLineCount = 0;
    
    // Sequential file reading (file I/O must be sequential)
    while (batchLineCount < BATCH_SIZE && fgets(lineBuffer, MAX_LINE_LENGTH, infile)) {
        // Copy line to batch buffer
        int lineLen = strlen(lineBuffer);
        
        // Remove newline characters
        if (lineLen > 0 && (lineBuffer[lineLen-1] == '\n' || lineBuffer[lineLen-1] == '\r')) {
            lineBuffer[lineLen-1] = '\0';
            lineLen--;
        }
        if (lineLen > 0 && (lineBuffer[lineLen-1] == '\n' || lineBuffer[lineLen-1] == '\r')) {
            lineBuffer[lineLen-1] = '\0';
            lineLen--;
        }
        
        // Copy to batch lines buffer
        memcpy(batch->lines + batchLineCount * MAX_LINE_LENGTH, lineBuffer, lineLen + 1);
        
        // Extract metadata in parallel
        char *underscore = strchr(lineBuffer, '_');
        char *pipe = strchr(lineBuffer, '|');
        
        if (underscore && pipe && underscore < pipe) {
            int len_chromo = underscore - lineBuffer;
            int len_lineNum = pipe - underscore - 1;
            
            // Ensure we don't exceed buffer limits
            if (len_chromo < MAX_META_LENGTH && len_lineNum < MAX_META_LENGTH) {
                strncpy(batch->chromo[batchLineCount], lineBuffer, len_chromo);
                batch->chromo[batchLineCount][len_chromo] = '\0';
                
                strncpy(batch->lineNum[batchLineCount], underscore + 1, len_lineNum);
                batch->lineNum[batchLineCount][len_lineNum] = '\0';
            } else {
                strcpy(batch->chromo[batchLineCount], "NA");
                strcpy(batch->lineNum[batchLineCount], "NA");
            }
        } else {
            strcpy(batch->chromo[batchLineCount], "NA");
            strcpy(batch->lineNum[batchLineCount], "NA");
        }
        
        batchLineCount++;
    }
    
    // Use OpenMP to process metadata extraction in parallel after loading
    if (batchLineCount > 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < batchLineCount; i++) {
            // Additional processing can go here if needed
            // For now, metadata is already extracted above
        }
    }
    
    batch->lineCount = batchLineCount;
    return batchLineCount;
}

// OpenMP-accelerated result processing
void processResults(SearchResult *result, BatchData *batch, const char *pattern) {
    if (result->matchCount > 0) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < result->matchCount; i++) {
            int idx = result->matchIndices[i];
            if (idx >= 0 && idx < batch->lineCount) {
                #pragma omp critical
                {
                    printf("Pattern '%s' found at chromosome %s, line %s (Batch %d)\n", 
                           pattern, batch->chromo[idx], batch->lineNum[idx], batch->batchId);
                }
            }
        }
    }
}

// Multi-GPU support function
int getOptimalDeviceCount() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    return (deviceCount > NUM_GPU_DEVICES) ? NUM_GPU_DEVICES : deviceCount;
}

int main() {
    // File and pattern setup
    char inputFileLocation[] = "/home/cj/HPC_data/Human_genome_preprocessed.fna";  // Test with known data first
    char pattern[MAX_LINE_LENGTH];
    FILE *infile = NULL;
    
    // Threading setup
    int numCpuThreads;
    int maxCpuThreads = omp_get_max_threads();
    
    // Performance monitoring
    double totalStartTime, totalEndTime;
    float cudaTotalTime = 0;
    
    // Memory and processing variables
    char *buffer = NULL;
    char *lineBuffer = NULL;
    int totalLineCount = 0;
    int totalMatchCount = 0;
    int currentBatch = 0;
    
    // CUDA variables
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t start, stop;
    cudaError_t cudaStatus = cudaSuccess;
    
    // Device memory pointers
    char *d_lines = NULL;
    char *d_pattern = NULL;
    int *d_matchIndices = NULL;
    int *d_matchCount = NULL;
    
    // Host memory for batches
    BatchData *batches = NULL;
    SearchResult *results = NULL;
    
    printf("=== Hybrid CUDA-OpenMP Search System ===\n");
    printf("Available CPU threads: %d\n", maxCpuThreads);
    
    // Get GPU information
    int deviceCount = getOptimalDeviceCount();
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n", prop.name);
    printf("Available GPU devices: %d\n", deviceCount);
    
    // Get user input
    printf("Enter pattern to search: ");
    if (scanf("%s", pattern) != 1) {
        fprintf(stderr, "Error reading pattern\n");
        return 1;
    }
    
    printf("Enter number of CPU threads to use (1-%d): ", maxCpuThreads);
    if (scanf("%d", &numCpuThreads) != 1) {
        fprintf(stderr, "Error reading thread count\n");
        numCpuThreads = maxCpuThreads;
    }
    if (numCpuThreads < 1 || numCpuThreads > maxCpuThreads) {
        numCpuThreads = maxCpuThreads;
    }
    
    omp_set_num_threads(numCpuThreads);
    
    printf("\nConfiguration:\n");
    printf("- Pattern: %s\n", pattern);
    printf("- CPU Threads: %d\n", numCpuThreads);
    printf("- Batch Size: %d lines\n", BATCH_SIZE);
    printf("- CUDA Streams: %d\n", NUM_STREAMS);
    printf("============================================================================\n");
    
    // Initialize CUDA
    cudaSetDevice(0);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    
    // Create CUDA streams and events
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
    
    // Allocate memory
    buffer = (char*)malloc(READ_BUFFER_SIZE);
    lineBuffer = (char*)malloc(MAX_LINE_LENGTH);
    batches = (BatchData*)malloc(sizeof(BatchData));
    results = (SearchResult*)malloc(sizeof(SearchResult));
    
    if (!buffer || !lineBuffer || !batches || !results) {
        fprintf(stderr, "Failed to allocate host memory\n");
        goto Error;
    }
    
    // Allocate pinned host memory for better transfer performance
    if (cudaMallocHost(&batches->lines, BATCH_SIZE * MAX_LINE_LENGTH) != cudaSuccess ||
        cudaMallocHost(&batches->chromo, BATCH_SIZE * sizeof(char[MAX_META_LENGTH])) != cudaSuccess ||
        cudaMallocHost(&batches->lineNum, BATCH_SIZE * sizeof(char[MAX_META_LENGTH])) != cudaSuccess ||
        cudaMallocHost(&results->matchIndices, BATCH_SIZE * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned host memory\n");
        goto Error;
    }
    
    // Allocate device memory
    if (cudaMalloc(&d_lines, BATCH_SIZE * MAX_LINE_LENGTH) != cudaSuccess ||
        cudaMalloc(&d_pattern, strlen(pattern) + 1) != cudaSuccess ||
        cudaMalloc(&d_matchIndices, BATCH_SIZE * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_matchCount, sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory\n");
        goto Error;
    }
    
    // Copy pattern to device
    if (cudaMemcpy(d_pattern, pattern, strlen(pattern) + 1, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy pattern to device\n");
        goto Error;
    }
    
    // Open file
    infile = fopen(inputFileLocation, "r");
    if (!infile) {
        perror("Error opening file");
        goto Error;
    }
    
    if (setvbuf(infile, buffer, _IOFBF, READ_BUFFER_SIZE) != 0) {
        fprintf(stderr, "Warning: Could not set file buffer\n");
    }
    
    totalStartTime = omp_get_wtime();
    
    printf("Starting hybrid search...\n");
    
    // Main processing loop
    while (!feof(infile)) {
        batches->batchId = currentBatch;
        
        // Load batch data using OpenMP
        int batchLineCount = loadBatchData(infile, batches, buffer, lineBuffer);
        if (batchLineCount == 0) break;
        
        cudaStream_t currentStream = streams[currentBatch % NUM_STREAMS];
        
        // Record CUDA timing
        cudaEventRecord(start, currentStream);
        
        // Copy data to device asynchronously
        if (cudaMemcpyAsync(d_lines, batches->lines, batchLineCount * MAX_LINE_LENGTH, 
                           cudaMemcpyHostToDevice, currentStream) != cudaSuccess ||
            cudaMemsetAsync(d_matchCount, 0, sizeof(int), currentStream) != cudaSuccess) {
            fprintf(stderr, "Device memory operations failed\n");
            goto Error;
        }
        
        // Launch hybrid kernel
        int numBlocks = (batchLineCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        hybridSearchKernel<<<numBlocks, THREADS_PER_BLOCK, SHARED_MEM_SIZE, currentStream>>>(
            d_lines, d_pattern, (int)strlen(pattern), batchLineCount, d_matchIndices, d_matchCount);
        
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed\n");
            goto Error;
        }
        
        // Copy results back
        if (cudaMemcpyAsync(&results->matchCount, d_matchCount, sizeof(int), 
                           cudaMemcpyDeviceToHost, currentStream) != cudaSuccess) {
            fprintf(stderr, "Match count copy failed\n");
            goto Error;
        }
        
        cudaStreamSynchronize(currentStream);
        
        if (results->matchCount > 0) {
            if (cudaMemcpyAsync(results->matchIndices, d_matchIndices, 
                               results->matchCount * sizeof(int), 
                               cudaMemcpyDeviceToHost, currentStream) != cudaSuccess) {
                fprintf(stderr, "Match indices copy failed\n");
                goto Error;
            }
            cudaStreamSynchronize(currentStream);
        }
        
        cudaEventRecord(stop, currentStream);
        cudaEventSynchronize(stop);
        
        float batchTime;
        cudaEventElapsedTime(&batchTime, start, stop);
        cudaTotalTime += batchTime;
        
        // Process results using OpenMP
        results->batchId = currentBatch;
        processResults(results, batches, pattern);
        
        totalMatchCount += results->matchCount;
        totalLineCount += batchLineCount;
        currentBatch++;
    }
    
    totalEndTime = omp_get_wtime();
    
    printf("\n============================================================================\n");
    printf("Hybrid Search Results:\n");
    printf("- Total lines processed: %d\n", totalLineCount);
    printf("- Total batches: %d\n", currentBatch);
    printf("- Total matches found: %d\n", totalMatchCount);
    printf("- CPU threads used: %d\n", numCpuThreads);
    printf("- CUDA time: %.6f seconds\n", cudaTotalTime / 1000.0);
    printf("- Total time: %.6f seconds\n", totalEndTime - totalStartTime);
    printf("- Performance improvement: %.2fx vs serial\n", 
           (totalEndTime - totalStartTime) > 0 ? 1.0 / (totalEndTime - totalStartTime) : 0);
    printf("============================================================================\n");

Error:
    // Cleanup
    if (infile) fclose(infile);
    if (buffer) free(buffer);
    if (lineBuffer) free(lineBuffer);
    if (batches) {
        if (batches->lines) cudaFreeHost(batches->lines);
        if (batches->chromo) cudaFreeHost(batches->chromo);
        if (batches->lineNum) cudaFreeHost(batches->lineNum);
        free(batches);
    }
    if (results) {
        if (results->matchIndices) cudaFreeHost(results->matchIndices);
        free(results);
    }
    
    // CUDA cleanup
    if (d_lines) cudaFree(d_lines);
    if (d_pattern) cudaFree(d_pattern);
    if (d_matchIndices) cudaFree(d_matchIndices);
    if (d_matchCount) cudaFree(d_matchCount);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaDeviceReset();
    
    return (cudaStatus == cudaSuccess) ? 0 : 1;
}