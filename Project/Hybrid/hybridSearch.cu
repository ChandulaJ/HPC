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
#define MAX_GPU_DEVICES 8  // Support for multiple GPU devices

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

// Structure for GPU device context
typedef struct {
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t start, stop;
    
    // Device memory pointers
    char *d_lines;
    char *d_pattern;
    int *d_matchIndices;
    int *d_matchCount;
    
    // Host memory
    BatchData *batches;
    SearchResult *results;
    
    // Processing stats
    float totalTime;
    int batchesProcessed;
    int totalMatches;
} GPUContext;

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

// Initialize GPU context
int initializeGPUContext(GPUContext *ctx, const char *pattern) {
    ctx->totalTime = 0;
    ctx->batchesProcessed = 0;
    ctx->totalMatches = 0;
    
    // Set device 0
    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "Failed to set device 0\n");
        return -1;
    }
    
    // Create streams and events
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (cudaStreamCreate(&ctx->streams[i]) != cudaSuccess) {
            fprintf(stderr, "Failed to create stream %d\n", i);
            return -1;
        }
    }
    
    if (cudaEventCreate(&ctx->start) != cudaSuccess || 
        cudaEventCreate(&ctx->stop) != cudaSuccess) {
        fprintf(stderr, "Failed to create events\n");
        return -1;
    }
    
    // Allocate device memory
    if (cudaMalloc(&ctx->d_lines, BATCH_SIZE * MAX_LINE_LENGTH) != cudaSuccess ||
        cudaMalloc(&ctx->d_pattern, strlen(pattern) + 1) != cudaSuccess ||
        cudaMalloc(&ctx->d_matchIndices, BATCH_SIZE * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&ctx->d_matchCount, sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return -1;
    }
    
    // Copy pattern to device
    if (cudaMemcpy(ctx->d_pattern, pattern, strlen(pattern) + 1, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy pattern to device\n");
        return -1;
    }
    
    // Allocate host memory
    ctx->batches = (BatchData*)malloc(sizeof(BatchData));
    ctx->results = (SearchResult*)malloc(sizeof(SearchResult));
    
    if (!ctx->batches || !ctx->results) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }
    
    // Allocate pinned host memory
    if (cudaMallocHost(&ctx->batches->lines, BATCH_SIZE * MAX_LINE_LENGTH) != cudaSuccess ||
        cudaMallocHost(&ctx->batches->chromo, BATCH_SIZE * sizeof(char[MAX_META_LENGTH])) != cudaSuccess ||
        cudaMallocHost(&ctx->batches->lineNum, BATCH_SIZE * sizeof(char[MAX_META_LENGTH])) != cudaSuccess ||
        cudaMallocHost(&ctx->results->matchIndices, BATCH_SIZE * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned host memory\n");
        return -1;
    }
    
    return 0;
}

// Cleanup GPU context
void cleanupGPUContext(GPUContext *ctx) {
    cudaSetDevice(0);
    
    // Free device memory
    if (ctx->d_lines) cudaFree(ctx->d_lines);
    if (ctx->d_pattern) cudaFree(ctx->d_pattern);
    if (ctx->d_matchIndices) cudaFree(ctx->d_matchIndices);
    if (ctx->d_matchCount) cudaFree(ctx->d_matchCount);
    
    // Destroy streams and events
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(ctx->streams[i]);
    }
    cudaEventDestroy(ctx->start);
    cudaEventDestroy(ctx->stop);
    
    // Free host memory
    if (ctx->batches) {
        if (ctx->batches->lines) cudaFreeHost(ctx->batches->lines);
        if (ctx->batches->chromo) cudaFreeHost(ctx->batches->chromo);
        if (ctx->batches->lineNum) cudaFreeHost(ctx->batches->lineNum);
        free(ctx->batches);
    }
    if (ctx->results) {
        if (ctx->results->matchIndices) cudaFreeHost(ctx->results->matchIndices);
        free(ctx->results);
    }
}

// Process file using OpenMP threads coordinating CUDA operations
void processFileWithOpenMP(GPUContext *ctx, FILE *file, const char *pattern, int numCpuThreads) {
    char *lineBuffer = (char*)malloc(MAX_LINE_LENGTH);
    
    if (!lineBuffer) {
        fprintf(stderr, "Failed to allocate line buffer\n");
        return;
    }
    
    int currentBatch = 0;
    
    printf("Processing file with %d OpenMP threads\n", numCpuThreads);
    
    while (!feof(file)) {
        ctx->batches->batchId = currentBatch;
        
        // Load batch data sequentially
        int batchLineCount = 0;
        while (batchLineCount < BATCH_SIZE && fgets(lineBuffer, MAX_LINE_LENGTH, file)) {
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
            memcpy(ctx->batches->lines + batchLineCount * MAX_LINE_LENGTH, lineBuffer, lineLen + 1);
            batchLineCount++;
        }
        
        if (batchLineCount == 0) break;
        
        // OpenMP parallel section
        #pragma omp parallel num_threads(numCpuThreads)
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            
            // Parallel metadata extraction
            int lines_per_thread = (batchLineCount + num_threads - 1) / num_threads;
            int start_idx = thread_id * lines_per_thread;
            int end_idx = (start_idx + lines_per_thread > batchLineCount) ? batchLineCount : start_idx + lines_per_thread;
            
            for (int i = start_idx; i < end_idx; i++) {
                char *line = ctx->batches->lines + i * MAX_LINE_LENGTH;
                char *underscore = strchr(line, '_');
                char *pipe = strchr(line, '|');
                
                if (underscore && pipe && underscore < pipe) {
                    int len_chromo = underscore - line;
                    int len_lineNum = pipe - underscore - 1;
                    
                    if (len_chromo < MAX_META_LENGTH && len_lineNum < MAX_META_LENGTH) {
                        strncpy(ctx->batches->chromo[i], line, len_chromo);
                        ctx->batches->chromo[i][len_chromo] = '\0';
                        
                        strncpy(ctx->batches->lineNum[i], underscore + 1, len_lineNum);
                        ctx->batches->lineNum[i][len_lineNum] = '\0';
                    } else {
                        strcpy(ctx->batches->chromo[i], "NA");
                        strcpy(ctx->batches->lineNum[i], "NA");
                    }
                } else {
                    strcpy(ctx->batches->chromo[i], "NA");
                    strcpy(ctx->batches->lineNum[i], "NA");
                }
            }
            
            #pragma omp barrier
            
            // Master thread handles CUDA operations
            #pragma omp master
            {
                ctx->batches->lineCount = batchLineCount;
                cudaStream_t currentStream = ctx->streams[currentBatch % NUM_STREAMS];
                
                cudaEventRecord(ctx->start, currentStream);
                
                if (cudaMemcpyAsync(ctx->d_lines, ctx->batches->lines, batchLineCount * MAX_LINE_LENGTH, 
                                   cudaMemcpyHostToDevice, currentStream) == cudaSuccess &&
                    cudaMemsetAsync(ctx->d_matchCount, 0, sizeof(int), currentStream) == cudaSuccess) {
                    
                    int numBlocks = (batchLineCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                    hybridSearchKernel<<<numBlocks, THREADS_PER_BLOCK, SHARED_MEM_SIZE, currentStream>>>(
                        ctx->d_lines, ctx->d_pattern, (int)strlen(pattern), batchLineCount, 
                        ctx->d_matchIndices, ctx->d_matchCount);
                    
                    if (cudaGetLastError() == cudaSuccess) {
                        if (cudaMemcpyAsync(&ctx->results->matchCount, ctx->d_matchCount, sizeof(int), 
                                           cudaMemcpyDeviceToHost, currentStream) == cudaSuccess) {
                            cudaStreamSynchronize(currentStream);
                            
                            if (ctx->results->matchCount > 0) {
                                cudaMemcpyAsync(ctx->results->matchIndices, ctx->d_matchIndices, 
                                               ctx->results->matchCount * sizeof(int), 
                                               cudaMemcpyDeviceToHost, currentStream);
                                cudaStreamSynchronize(currentStream);
                            }
                            
                            cudaEventRecord(ctx->stop, currentStream);
                            cudaEventSynchronize(ctx->stop);
                            
                            float batchTime;
                            cudaEventElapsedTime(&batchTime, ctx->start, ctx->stop);
                            ctx->totalTime += batchTime;
                            ctx->totalMatches += ctx->results->matchCount;
                        }
                    }
                }
            }
            
            #pragma omp barrier
            
            // Parallel result processing - prepare formatted results
            if (ctx->results->matchCount > 0) {
                // Allocate local storage for formatted results
                static char **formattedResults = NULL;
                static int *validResults = NULL;
                
                #pragma omp master
                {
                    formattedResults = (char**)malloc(ctx->results->matchCount * sizeof(char*));
                    validResults = (int*)malloc(ctx->results->matchCount * sizeof(int));
                    for (int i = 0; i < ctx->results->matchCount; i++) {
                        formattedResults[i] = (char*)malloc(512 * sizeof(char)); // 512 chars per result
                        validResults[i] = 0; // Initialize as invalid
                    }
                }
                
                #pragma omp barrier
                
                // Parallel formatting of results
                #pragma omp for schedule(dynamic, 4)
                for (int i = 0; i < ctx->results->matchCount; i++) {
                    int idx = ctx->results->matchIndices[i];
                    if (idx >= 0 && idx < ctx->batches->lineCount) {
                        snprintf(formattedResults[i], 512,
                                "Thread %d - Pattern '%s' found at chromosome %s, line %s (Batch %d)",
                                thread_id, pattern, ctx->batches->chromo[idx], 
                                ctx->batches->lineNum[idx], ctx->batches->batchId);
                        validResults[i] = 1; // Mark as valid
                    }
                }
                
                #pragma omp barrier
                
                // Sequential output by master thread
                #pragma omp master
                {
                    for (int i = 0; i < ctx->results->matchCount; i++) {
                        if (validResults[i]) {
                            printf("%s\n", formattedResults[i]);
                        }
                    }
                    
                    // Cleanup
                    for (int i = 0; i < ctx->results->matchCount; i++) {
                        free(formattedResults[i]);
                    }
                    free(formattedResults);
                    free(validResults);
                    formattedResults = NULL;
                    validResults = NULL;
                }
            }
        }
        
        ctx->batchesProcessed++;
        currentBatch++;
    }
    
    free(lineBuffer);
}

int main() {
    // File and pattern setup
    char inputFileLocation[] = "/home/cj/HPC_data/Human_genome_preprocessed.fna";
    char pattern[MAX_LINE_LENGTH];
    FILE *infile = NULL;
    
    // Threading setup
    int numCpuThreads;
    int maxCpuThreads = omp_get_max_threads();
    
    // Performance monitoring
    double totalStartTime, totalEndTime;
    
    printf("=== Hybrid CUDA-OpenMP Search System ===\n");
    printf("Available CPU threads: %d\n", maxCpuThreads);
    
    // Check for CUDA device
    int deviceCount;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n", prop.name);
    
    // Get user input
    printf("Enter pattern to search: ");
    if (scanf("%s", pattern) != 1) {
        fprintf(stderr, "Error reading pattern\n");
        return 1;
    }
    
    printf("Enter number of OpenMP threads (1-%d): ", maxCpuThreads);
    if (scanf("%d", &numCpuThreads) != 1) {
        fprintf(stderr, "Error reading thread count\n");
        numCpuThreads = maxCpuThreads;
    }
    if (numCpuThreads < 1 || numCpuThreads > maxCpuThreads) {
        numCpuThreads = maxCpuThreads;
    }
    
    printf("\nConfiguration:\n");
    printf("- Pattern: %s\n", pattern);
    printf("- GPU Device: %s\n", prop.name);
    printf("- OpenMP Threads: %d\n", numCpuThreads);
    printf("- Batch Size: %d lines\n", BATCH_SIZE);
    printf("- CUDA Streams: %d\n", NUM_STREAMS);
    printf("============================================================================\n");
    
    // Open file
    infile = fopen(inputFileLocation, "r");
    if (!infile) {
        perror("Error opening file");
        return 1;
    }
    
    // Initialize GPU context
    GPUContext gpuContext;
    if (initializeGPUContext(&gpuContext, pattern) != 0) {
        fprintf(stderr, "Failed to initialize GPU\n");
        fclose(infile);
        return 1;
    }
    
    printf("Starting hybrid CUDA-OpenMP search...\n");
    totalStartTime = omp_get_wtime();
    
    // Process file with OpenMP coordinating CUDA operations
    processFileWithOpenMP(&gpuContext, infile, pattern, numCpuThreads);
    
    totalEndTime = omp_get_wtime();
    
    // Display results
    printf("\n============================================================================\n");
    printf("Hybrid CUDA-OpenMP Search Results:\n");
    printf("- Batches processed: %d\n", gpuContext.batchesProcessed);
    printf("- Total matches found: %d\n", gpuContext.totalMatches);
    printf("- GPU time: %.6f seconds\n", gpuContext.totalTime / 1000.0);
    printf("- Total wall time: %.6f seconds\n", totalEndTime - totalStartTime);
    printf("- OpenMP threads used: %d\n", numCpuThreads);
    printf("- Efficiency ratio: %.2f%%\n", 
           ((gpuContext.totalTime / 1000.0) / (totalEndTime - totalStartTime)) * 100.0);
    printf("============================================================================\n");
    
    // Cleanup
    cleanupGPUContext(&gpuContext);
    fclose(infile);
    
    return 0;
}
