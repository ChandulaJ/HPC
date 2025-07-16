#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_PATTERN_LENGTH 1024
#define THREADS_PER_BLOCK 256
#define CHUNK_SIZE (1024 * 1024)
#define MAX_MATCHES_PER_CHUNK 10000
#define INITIAL_MATCHES_CAPACITY 1000

typedef struct {
    int position;
    char line_info[200];
} Match;

typedef struct {
    char* text_chunk;
    int chunk_size;
    int chunk_offset;
    Match* matches;
    int match_count;
} ChunkData;

__global__ void searchKernel(const char* text, const char* pattern, 
                           int text_size, int pattern_length, 
                           int* match_positions, int* match_count, int chunk_offset) {
    __shared__ char shared_pattern[MAX_PATTERN_LENGTH];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load pattern into shared memory cooperatively
    if (tid < pattern_length) {
        shared_pattern[tid] = pattern[tid];
    }
    __syncthreads();
    
    if (idx < text_size) {
        // Find line boundaries more efficiently
        int line_start = idx;
        while (line_start > 0 && text[line_start - 1] != '\n') {
            line_start--;
        }
        
        // Only process if we're at the start of a line
        if (idx == line_start) {
            int line_end = idx;
            while (line_end < text_size && text[line_end] != '\n' && text[line_end] != '\0') {
                line_end++;
            }
            
            int line_length = line_end - line_start;
            
            if (line_length >= pattern_length) {
                // Use shared memory pattern for faster comparison
                for (int i = 0; i <= line_length - pattern_length; i++) {
                    bool match = true;
                    
                    // Unroll small patterns for better performance
                    if (pattern_length <= 8) {
                        for (int j = 0; j < pattern_length; j++) {
                            if (text[line_start + i + j] != shared_pattern[j]) {
                                match = false;
                                break;
                            }
                        }
                    } else {
                        // Use vectorized comparison for longer patterns
                        for (int j = 0; j < pattern_length; j += 4) {
                            int remaining = min(4, pattern_length - j);
                            for (int k = 0; k < remaining; k++) {
                                if (text[line_start + i + j + k] != shared_pattern[j + k]) {
                                    match = false;
                                    break;
                                }
                            }
                            if (!match) break;
                        }
                    }
                    
                    if (match) {
                        int pos = atomicAdd(match_count, 1);
                        if (pos < MAX_MATCHES_PER_CHUNK) {
                            match_positions[pos] = line_start + chunk_offset;
                        }
                        break;
                    }
                }
            }
        }
    }
}

char* readTextChunk(FILE* file, int chunk_size, int* actual_size) {
    // Use posix_memalign for better compatibility
    char* chunk = NULL;
    if (posix_memalign((void**)&chunk, 64, chunk_size + 64) != 0) {
        // Fallback to regular malloc if posix_memalign fails
        chunk = (char*)malloc(chunk_size + 64);
    }
    if (!chunk) return NULL;
    
    *actual_size = fread(chunk, 1, chunk_size, file);
    chunk[*actual_size] = '\0';
    return chunk;
}

long getFileSize(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return -1;
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    return size;
}

void cleanupCudaMemory(char* d_text, char* d_pattern, int* d_match_positions, int* d_match_count) {
    if (d_text) cudaFree(d_text);
    if (d_pattern) cudaFree(d_pattern);
    if (d_match_positions) cudaFree(d_match_positions);
    if (d_match_count) cudaFree(d_match_count);
}

int processChunkWithCUDA(ChunkData* chunk_data, const char* pattern, int pattern_length) {
    char* d_text = NULL;
    char* d_pattern = NULL;
    int* d_match_positions = NULL;
    int* d_match_count = NULL;
    int h_match_count = 0;
    
    // Use CUDA streams for better performance
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        return 0;
    }
    
    // Allocate device memory with error checking
    if (cudaMalloc(&d_text, chunk_data->chunk_size) != cudaSuccess ||
        cudaMalloc(&d_pattern, pattern_length) != cudaSuccess ||
        cudaMalloc(&d_match_positions, MAX_MATCHES_PER_CHUNK * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_match_count, sizeof(int)) != cudaSuccess) {
        cleanupCudaMemory(d_text, d_pattern, d_match_positions, d_match_count);
        cudaStreamDestroy(stream);
        return 0;
    }
    
    // Asynchronous memory transfers
    cudaMemcpyAsync(d_text, chunk_data->text_chunk, chunk_data->chunk_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pattern, pattern, pattern_length, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_match_count, 0, sizeof(int), stream);
    
    // Optimize grid size based on occupancy
    int min_grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, searchKernel, 0, 0);
    
    // Use optimal block size but cap it at our defined maximum
    block_size = min(block_size, THREADS_PER_BLOCK);
    int num_blocks = (chunk_data->chunk_size + block_size - 1) / block_size;
    
    // Launch kernel with stream
    searchKernel<<<num_blocks, block_size, 0, stream>>>(
        d_text, d_pattern, chunk_data->chunk_size, pattern_length,
        d_match_positions, d_match_count, chunk_data->chunk_offset);
    
    // Check for errors and synchronize
    if (cudaGetLastError() != cudaSuccess) {
        cleanupCudaMemory(d_text, d_pattern, d_match_positions, d_match_count);
        cudaStreamDestroy(stream);
        return 0;
    }
    
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(&h_match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (h_match_count > 0) {
        int* h_match_positions = (int*)malloc(h_match_count * sizeof(int));
        cudaMemcpyAsync(h_match_positions, d_match_positions, 
                       h_match_count * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        chunk_data->matches = (Match*)malloc(h_match_count * sizeof(Match));
        chunk_data->match_count = h_match_count;
        
        // Optimize line info extraction
        for (int i = 0; i < h_match_count; i++) {
            chunk_data->matches[i].position = h_match_positions[i];
            
            int line_pos = h_match_positions[i] - chunk_data->chunk_offset;
            int line_start = line_pos;
            
            // Find line start more efficiently
            while (line_start > 0 && chunk_data->text_chunk[line_start - 1] != '\n') {
                line_start--;
            }
            
            // Extract line info with early termination
            int extract_len = 0;
            char* src = &chunk_data->text_chunk[line_start];
            char* dst = chunk_data->matches[i].line_info;
            
            while (extract_len < 199 && 
                   (line_start + extract_len) < chunk_data->chunk_size &&
                   src[extract_len] != '\n' && src[extract_len] != '\0') {
                dst[extract_len] = src[extract_len];
                extract_len++;
                
                // Early termination after pipe
                if (src[extract_len - 1] == '|') {
                    int extra = min(20, min(199 - extract_len, chunk_data->chunk_size - line_start - extract_len));
                    for (int j = 0; j < extra && src[extract_len + j] != '\n' && src[extract_len + j] != '\0'; j++) {
                        dst[extract_len + j] = src[extract_len + j];
                        extract_len++;
                    }
                    break;
                }
            }
            dst[extract_len] = '\0';
        }
        
        free(h_match_positions);
    } else {
        chunk_data->matches = NULL;
        chunk_data->match_count = 0;
    }
    
    cleanupCudaMemory(d_text, d_pattern, d_match_positions, d_match_count);
    cudaStreamDestroy(stream);
    return h_match_count;
}

int compareMatches(const void* a, const void* b) {
    const Match* ma = (const Match*)a;
    const Match* mb = (const Match*)b;
    return (ma->position > mb->position) - (ma->position < mb->position);
}

void printMatchResult(const char* line) {
    const char *underscore = strchr(line, '_');
    const char *pipe = strchr(line, '|');
    
    if (underscore && pipe && underscore < pipe) {
        printf("Pattern found at chromosome %.*s, at line %.*s\n", 
               (int)(underscore - line), line,
               (int)(pipe - underscore - 1), underscore + 1);
    }
}

int main() {
    char text_filename[] = "/home/cj/HPC_data/Human_genome_preprocessed.fna";
    char input_pattern[MAX_PATTERN_LENGTH];
    char* pattern;
    long file_size;
    int pattern_length, num_threads;
    
    printf("=== Hybrid OpenMP-CUDA Pattern Search ===\n");
    printf("Text file: %s\n", text_filename);
    
    printf("Enter pattern to search: ");
    scanf("%s", input_pattern);
    
    printf("Enter number of OpenMP threads: ");
    scanf("%d", &num_threads);
    
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
        printf("Using default number of threads: %d\n", num_threads);
    }
    
    file_size = getFileSize(text_filename);
    if (file_size <= 0) {
        printf("Error: Cannot open or read file %s\n", text_filename);
        return 1;
    }
    
    pattern_length = strlen(input_pattern);
    pattern = (char*)malloc(pattern_length + 1);
    if (!pattern) {
        printf("Error: Cannot allocate memory for pattern\n");
        return 1;
    }
    strcpy(pattern, input_pattern);
    
    printf("\nConfiguration:\n");
    printf("- File size: %ld bytes\n", file_size);
    printf("- Pattern: '%s' (length: %d)\n", pattern, pattern_length);
    printf("- OpenMP threads: %d\n", num_threads);
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("Error: No CUDA devices found\n");
        free(pattern);
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("- GPU: %s\n", prop.name);
    
    int num_chunks = (file_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    printf("- Number of chunks: %d\n", num_chunks);
    printf("\nStarting hybrid search...\n");
    
    FILE* file = fopen(text_filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", text_filename);
        free(pattern);
        return 1;
    }
    
    int total_matches = 0;
    int matches_capacity = INITIAL_MATCHES_CAPACITY;
    Match* all_matches = (Match*)malloc(matches_capacity * sizeof(Match));
    if (!all_matches) {
        printf("Error: Cannot allocate initial memory for matches\n");
        free(pattern);
        return 1;
    }
    
    // Pre-read all chunks into memory for parallel processing
    char** all_chunks = (char**)calloc(num_chunks, sizeof(char*));
    int* chunk_sizes = (int*)calloc(num_chunks, sizeof(int));
    if (!all_chunks || !chunk_sizes) {
        printf("Error: Cannot allocate memory for chunk management\n");
        free(pattern);
        free(all_matches);
        if (all_chunks) free(all_chunks);
        if (chunk_sizes) free(chunk_sizes);
        return 1;
    }
    int actual_chunks = 0;
    
    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        all_chunks[chunk_id] = readTextChunk(file, CHUNK_SIZE, &chunk_sizes[chunk_id]);
        if (!all_chunks[chunk_id] || chunk_sizes[chunk_id] == 0) break;
        actual_chunks++;
    }
    fclose(file);
    
    // Thread-local storage for matches
    ChunkData* chunk_results = (ChunkData*)calloc(actual_chunks, sizeof(ChunkData));
    if (!chunk_results) {
        printf("Error: Cannot allocate memory for chunk results\n");
        for (int i = 0; i < actual_chunks; i++) {
            if (all_chunks[i]) free(all_chunks[i]);
        }
        free(all_chunks);
        free(chunk_sizes);
        free(all_matches);
        free(pattern);
        return 1;
    }
    
    double start_time = omp_get_wtime();
    
    // Set OpenMP thread count
    omp_set_num_threads(num_threads);
    
    // Parallel processing of chunks
    #pragma omp parallel for schedule(dynamic)
    for (int chunk_id = 0; chunk_id < actual_chunks; chunk_id++) {
        chunk_results[chunk_id].text_chunk = all_chunks[chunk_id];
        chunk_results[chunk_id].chunk_size = chunk_sizes[chunk_id];
        chunk_results[chunk_id].chunk_offset = chunk_id * CHUNK_SIZE;
        chunk_results[chunk_id].matches = NULL;
        chunk_results[chunk_id].match_count = 0;
        
        processChunkWithCUDA(&chunk_results[chunk_id], pattern, pattern_length);
    }
    
    // Ensure all GPU operations are complete before processing results
    cudaDeviceSynchronize();
    
    // Collect results from all threads with optimized memory management
    for (int chunk_id = 0; chunk_id < actual_chunks; chunk_id++) {
        if (chunk_results[chunk_id].match_count > 0) {
            // Expand capacity if needed
            while (total_matches + chunk_results[chunk_id].match_count > matches_capacity) {
                matches_capacity *= 2;
                Match* temp = (Match*)realloc(all_matches, matches_capacity * sizeof(Match));
                if (!temp) {
                    printf("Error: Cannot allocate memory for matches\n");
                    // Clean up allocated memory before exit
                    if (chunk_results[chunk_id].matches) free(chunk_results[chunk_id].matches);
                    goto cleanup;
                }
                all_matches = temp;
            }
            
            // Copy matches efficiently
            memcpy(&all_matches[total_matches], chunk_results[chunk_id].matches, 
                   chunk_results[chunk_id].match_count * sizeof(Match));
            total_matches += chunk_results[chunk_id].match_count;
            free(chunk_results[chunk_id].matches);
            chunk_results[chunk_id].matches = NULL; // Prevent double free
        }
    }
    
    // Clean up chunks after all processing is done
    for (int chunk_id = 0; chunk_id < actual_chunks; chunk_id++) {
        if (all_chunks[chunk_id]) {
            free(all_chunks[chunk_id]);
            all_chunks[chunk_id] = NULL;
        }
    }
    
    cleanup:
    if (all_chunks) free(all_chunks);
    if (chunk_sizes) free(chunk_sizes);
    if (chunk_results) free(chunk_results);
    
    
    double end_time = omp_get_wtime();
    
    printf("\nSearch completed in %.6f seconds\n", end_time - start_time);
    
    if (total_matches > 0) {
        printf("============================================================================\n");
        
        // Use quicksort instead of bubble sort for better performance
        qsort(all_matches, total_matches, sizeof(Match), compareMatches);
        
        for (int i = 0; i < total_matches; i++) {
            printMatchResult(all_matches[i].line_info);
        }
        
        printf("============================================================================\n");
    } else {
        printf("============================================================================\n");
    }
    
    printf("Total matches found: %d\n", total_matches);
    printf("Time taken for hybrid OpenMP-CUDA search: %.6f seconds\n", end_time - start_time);
    printf("Processing rate: %.2f MB/s\n", (double)file_size / (1024 * 1024) / (end_time - start_time));
    printf("Chunks processed in parallel: %d\n", actual_chunks);
    
    // Clean up remaining memory
    if (all_matches) {
        free(all_matches);
        all_matches = NULL;
    }
    if (pattern) {
        free(pattern);
        pattern = NULL;
    }
    
    return 0;
}
