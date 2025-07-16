#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Optimized constants for hybrid processing
#define MAX_LINE_LENGTH 105
#define BATCH_SIZE 131072  // Increased batch size for better GPU utilization
#define MAX_META_LENGTH 32
#define READ_BUFFER_SIZE (64 * 1024 * 1024)  // Larger buffer for faster I/O
#define THREADS_PER_BLOCK 1024  // Increased for better GPU occupancy
#define SHARED_MEM_SIZE 512  // Larger shared memory
#define NUM_STREAMS 8  // More streams for better overlap
#define MAX_GPU_DEVICES 8

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

// Enhanced CUDA kernel with optimized pattern matching
__global__ void hybridSearchKernel(
    const char *d_lines,
    const char *d_pattern,
    int pattSize,
    int lineCount,
    int *d_matchIndices,
    int *d_matchCount
) {
    __shared__ char sharedPattern[SHARED_MEM_SIZE];
    __shared__ int localMatches[1024];  // Local match buffer
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int globalIdx = bid * blockDim.x + tid;
    
    // Cooperative loading of pattern into shared memory
    if (tid < pattSize && tid < SHARED_MEM_SIZE) {
        sharedPattern[tid] = d_pattern[tid];
    }
    
    // Initialize local match buffer
    if (tid < 1024) {
        localMatches[tid] = -1;
    }
    
    __syncthreads();
    
    // Each thread processes one line
    if (globalIdx >= lineCount) return;
    
    const char *line = d_lines + globalIdx * MAX_LINE_LENGTH;
    int lineLen = 0;
    
    // Optimized line length calculation with early termination
    while (lineLen < MAX_LINE_LENGTH && line[lineLen] > '\r') {
        lineLen++;
    }
    
    // Boyer-Moore-style optimized pattern matching
    bool found = false;
    if (lineLen >= pattSize) {
        for (int i = 0; i <= lineLen - pattSize && !found; i++) {
            // Quick first character check
            if (line[i] == sharedPattern[0]) {
                bool matched = true;
                // Unrolled comparison for small patterns
                if (pattSize <= 4) {
                    for (int j = 1; j < pattSize; j++) {
                        if (line[i + j] != sharedPattern[j]) {
                            matched = false;
                            break;
                        }
                    }
                } else {
                    // Standard comparison for longer patterns
                    for (int j = 1; j < pattSize; j++) {
                        if (line[i + j] != sharedPattern[j]) {
                            matched = false;
                            break;
                        }
                    }
                }
                
                if (matched) {
                    found = true;
                    // Store in local buffer first
                    localMatches[tid] = globalIdx;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Coalesced global memory writes
    if (tid == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            if (localMatches[i] != -1) {
                int pos = atomicAdd(d_matchCount, 1);
                if (pos < BATCH_SIZE) {
                    d_matchIndices[pos] = localMatches[i];
                }
            }
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

// Enhanced file processing with prefetching and optimized I/O
void processFileWithOpenMP(GPUContext *ctx, FILE *file, const char *pattern, int numCpuThreads) {
    // Allocate larger buffers for better performance
    char *lineBuffer = (char*)malloc(MAX_LINE_LENGTH);
    char *readBuffer = (char*)malloc(READ_BUFFER_SIZE);
    
    if (!lineBuffer || !readBuffer) {
        fprintf(stderr, "Failed to allocate buffers\n");
        return;
    }
    
    // Set larger file buffer for faster I/O
    if (setvbuf(file, readBuffer, _IOFBF, READ_BUFFER_SIZE) != 0) {
        fprintf(stderr, "Warning: Could not set file buffer\n");
    }
    
    int currentBatch = 0;
    
    printf("Processing file with %d OpenMP threads (Enhanced Mode)\n", numCpuThreads);
    
    while (!feof(file)) {
        ctx->batches->batchId = currentBatch;
        
        // Optimized batch loading with prefetch hints
        int batchLineCount = 0;
        while (batchLineCount < BATCH_SIZE && fgets(lineBuffer, MAX_LINE_LENGTH, file)) {
            int lineLen = strlen(lineBuffer);
            
            // Optimized newline removal
            if (lineLen > 0) {
                char *end = lineBuffer + lineLen - 1;
                while (end >= lineBuffer && (*end == '\n' || *end == '\r')) {
                    *end-- = '\0';
                    lineLen--;
                }
            }
            
            // Fast memory copy with prefetch
            char *dest = ctx->batches->lines + batchLineCount * MAX_LINE_LENGTH;
            __builtin_prefetch(dest, 1, 1);  // Prefetch for write
            memcpy(dest, lineBuffer, lineLen + 1);
            batchLineCount++;
        }
        
        if (batchLineCount == 0) break;
        
        // Enhanced OpenMP parallel section with better load balancing
        #pragma omp parallel num_threads(numCpuThreads)
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            
            // Optimized work distribution with cache-friendly access
            int chunk_size = (batchLineCount + num_threads - 1) / num_threads;
            int start_idx = thread_id * chunk_size;
            int end_idx = (start_idx + chunk_size > batchLineCount) ? batchLineCount : start_idx + chunk_size;
            
            // Parallel metadata extraction with vectorized operations
            for (int i = start_idx; i < end_idx; i++) {
                char *line = ctx->batches->lines + i * MAX_LINE_LENGTH;
                
                // Fast character search using optimized string functions
                char *underscore = strchr(line, '_');
                char *pipe = strchr(line, '|');
                
                if (underscore && pipe && underscore < pipe) {
                    size_t len_chromo = underscore - line;
                    size_t len_lineNum = pipe - underscore - 1;
                    
                    if (len_chromo < MAX_META_LENGTH && len_lineNum < MAX_META_LENGTH) {
                        // Use memcpy for better performance
                        memcpy(ctx->batches->chromo[i], line, len_chromo);
                        ctx->batches->chromo[i][len_chromo] = '\0';
                        
                        memcpy(ctx->batches->lineNum[i], underscore + 1, len_lineNum);
                        ctx->batches->lineNum[i][len_lineNum] = '\0';
                    } else {
                        // Fast constant assignment
                        ctx->batches->chromo[i][0] = 'N'; ctx->batches->chromo[i][1] = 'A'; ctx->batches->chromo[i][2] = '\0';
                        ctx->batches->lineNum[i][0] = 'N'; ctx->batches->lineNum[i][1] = 'A'; ctx->batches->lineNum[i][2] = '\0';
                    }
                } else {
                    ctx->batches->chromo[i][0] = 'N'; ctx->batches->chromo[i][1] = 'A'; ctx->batches->chromo[i][2] = '\0';
                    ctx->batches->lineNum[i][0] = 'N'; ctx->batches->lineNum[i][1] = 'A'; ctx->batches->lineNum[i][2] = '\0';
                }
            }
            
            #pragma omp barrier
            
            // Master thread handles optimized CUDA operations
            #pragma omp master
            {
                ctx->batches->lineCount = batchLineCount;
                cudaStream_t currentStream = ctx->streams[currentBatch % NUM_STREAMS];
                
                // Use CUDA events for precise timing
                cudaEventRecord(ctx->start, currentStream);
                
                // Optimized memory transfers with better alignment
                size_t dataSize = batchLineCount * MAX_LINE_LENGTH;
                if (cudaMemcpyAsync(ctx->d_lines, ctx->batches->lines, dataSize, 
                                   cudaMemcpyHostToDevice, currentStream) == cudaSuccess &&
                    cudaMemsetAsync(ctx->d_matchCount, 0, sizeof(int), currentStream) == cudaSuccess) {
                    
                    // Optimized kernel launch parameters
                    int numBlocks = (batchLineCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                    size_t sharedMemSize = SHARED_MEM_SIZE + 1024 * sizeof(int);  // Pattern + local buffer
                    
                    hybridSearchKernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize, currentStream>>>(
                        ctx->d_lines, ctx->d_pattern, (int)strlen(pattern), batchLineCount, 
                        ctx->d_matchIndices, ctx->d_matchCount);
                    
                    // Check for kernel launch errors
                    cudaError_t kernelError = cudaGetLastError();
                    if (kernelError == cudaSuccess) {
                        // Async memory transfer back
                        if (cudaMemcpyAsync(&ctx->results->matchCount, ctx->d_matchCount, sizeof(int), 
                                           cudaMemcpyDeviceToHost, currentStream) == cudaSuccess) {
                            cudaStreamSynchronize(currentStream);
                            
                            // Only transfer match indices if there are matches
                            if (ctx->results->matchCount > 0) {
                                size_t matchSize = ctx->results->matchCount * sizeof(int);
                                cudaMemcpyAsync(ctx->results->matchIndices, ctx->d_matchIndices, 
                                               matchSize, cudaMemcpyDeviceToHost, currentStream);
                                cudaStreamSynchronize(currentStream);
                            }
                            
                            // Record timing
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
            
            // Optimized parallel result processing
            if (ctx->results->matchCount > 0) {
                static char **formattedResults = NULL;
                static int *validResults = NULL;
                
                #pragma omp master
                {
                    // Use aligned allocation for better cache performance
                    size_t resultCount = ctx->results->matchCount;
                    formattedResults = (char**)aligned_alloc(64, resultCount * sizeof(char*));
                    validResults = (int*)aligned_alloc(64, resultCount * sizeof(int));
                    
                    for (int i = 0; i < resultCount; i++) {
                        formattedResults[i] = (char*)aligned_alloc(64, 256 * sizeof(char)); // Reduced size
                        validResults[i] = 0;
                    }
                }
                
                #pragma omp barrier
                
                // Parallel formatting with static scheduling for better cache locality
                #pragma omp for schedule(static, 16)
                for (int i = 0; i < ctx->results->matchCount; i++) {
                    int idx = ctx->results->matchIndices[i];
                    if (idx >= 0 && idx < ctx->batches->lineCount) {
                        // Simplified output format for speed
                        snprintf(formattedResults[i], 256,
                                "T%d: %s found at %s:%s (B%d)",
                                thread_id, pattern, ctx->batches->chromo[idx], 
                                ctx->batches->lineNum[idx], ctx->batches->batchId);
                        validResults[i] = 1;
                    }
                }
                
                #pragma omp barrier
                
                // Fast sequential output
                #pragma omp master
                {
                    for (int i = 0; i < ctx->results->matchCount; i++) {
                        if (validResults[i]) {
                            puts(formattedResults[i]);  // Faster than printf
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
        
        // Periodically yield to prevent thread starvation
        if (currentBatch % 100 == 0) {
            #pragma omp flush
        }
    }
    
    free(lineBuffer);
    free(readBuffer);
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
