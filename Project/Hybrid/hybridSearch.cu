#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constants for hybrid processing
#define MAX_TEXT_SIZE (1024 * 1024 * 1024)  // 1GB max text size
#define MAX_PATTERN_LENGTH 1024
#define THREADS_PER_BLOCK 256
#define CHUNK_SIZE (1024 * 1024)  // 1MB chunks for OpenMP threads
#define MAX_MATCHES_PER_CHUNK 10000

// Structure to hold match information with line content
typedef struct {
    int position;
    int thread_id;
    char line_info[200];  // Store the front part of the line for chromosome/line number extraction
} Match;

// Structure to hold chunk processing data
typedef struct {
    char* text_chunk;
    int chunk_size;
    int chunk_offset;
    Match* matches;
    int match_count;
    int thread_id;
} ChunkData;

// CUDA kernel for pattern searching within a chunk (line-based)
__global__ void searchKernel(const char* text, const char* pattern, 
                           int text_size, int pattern_length, 
                           int* match_positions, int* match_count, int chunk_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Find line boundaries and search within each line
    if (idx < text_size) {
        // Find the start of the current line
        int line_start = idx;
        while (line_start > 0 && text[line_start - 1] != '\n') {
            line_start--;
        }
        
        // Find the end of the current line
        int line_end = idx;
        while (line_end < text_size && text[line_end] != '\n' && text[line_end] != '\0') {
            line_end++;
        }
        
        int line_length = line_end - line_start;
        
        // Only process if we're at the start of a line to avoid duplicate processing
        if (idx == line_start && line_length >= pattern_length) {
            // Search for pattern in this line
            for (int i = 0; i <= line_length - pattern_length; i++) {
                bool match = true;
                
                // Check if pattern matches at this position
                for (int j = 0; j < pattern_length; j++) {
                    if (text[line_start + i + j] != pattern[j]) {
                        match = false;
                        break;
                    }
                }
                
                // If match found, record the line start position
                if (match) {
                    int pos = atomicAdd(match_count, 1);
                    if (pos < MAX_MATCHES_PER_CHUNK) {
                        match_positions[pos] = line_start + chunk_offset;
                    }
                    break; // Only record one match per line
                }
            }
        }
    }
}

// Function to read text from file in streaming chunks
char* readTextChunk(FILE* file, int chunk_size, int* actual_size) {
    char* chunk = (char*)malloc(chunk_size + 1);
    if (!chunk) {
        printf("Error: Cannot allocate memory for text chunk\n");
        return NULL;
    }
    
    *actual_size = fread(chunk, 1, chunk_size, file);
    chunk[*actual_size] = '\0';
    
    return chunk;
}

// Function to get file size without loading entire file
long getFileSize(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        return -1;
    }
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    
    return size;
}

// Function to process a chunk using CUDA
int processChunkWithCUDA(ChunkData* chunk_data, const char* pattern, int pattern_length) {
    // Device memory pointers
    char* d_text = NULL;
    char* d_pattern = NULL;
    int* d_match_positions = NULL;
    int* d_match_count = NULL;
    int h_match_count = 0;
    
    // Allocate device memory
    cudaError_t err;
    
    err = cudaMalloc(&d_text, chunk_data->chunk_size);
    if (err != cudaSuccess) {
        printf("Thread %d: Error allocating device memory for text: %s\n", 
               chunk_data->thread_id, cudaGetErrorString(err));
        return 0;
    }
    
    err = cudaMalloc(&d_pattern, pattern_length);
    if (err != cudaSuccess) {
        printf("Thread %d: Error allocating device memory for pattern: %s\n", 
               chunk_data->thread_id, cudaGetErrorString(err));
        cudaFree(d_text);
        return 0;
    }
    
    err = cudaMalloc(&d_match_positions, MAX_MATCHES_PER_CHUNK * sizeof(int));
    if (err != cudaSuccess) {
        printf("Thread %d: Error allocating device memory for match positions: %s\n", 
               chunk_data->thread_id, cudaGetErrorString(err));
        cudaFree(d_text);
        cudaFree(d_pattern);
        return 0;
    }
    
    err = cudaMalloc(&d_match_count, sizeof(int));
    if (err != cudaSuccess) {
        printf("Thread %d: Error allocating device memory for match count: %s\n", 
               chunk_data->thread_id, cudaGetErrorString(err));
        cudaFree(d_text);
        cudaFree(d_pattern);
        cudaFree(d_match_positions);
        return 0;
    }
    
    // Copy data to device
    cudaMemcpy(d_text, chunk_data->text_chunk, chunk_data->chunk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern, pattern_length, cudaMemcpyHostToDevice);
    cudaMemset(d_match_count, 0, sizeof(int));
    
    // Calculate grid and block dimensions
    int num_blocks = (chunk_data->chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Launch kernel
    searchKernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_text, d_pattern, chunk_data->chunk_size, pattern_length,
        d_match_positions, d_match_count, chunk_data->chunk_offset);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d: Kernel launch error: %s\n", 
               chunk_data->thread_id, cudaGetErrorString(err));
        cudaFree(d_text);
        cudaFree(d_pattern);
        cudaFree(d_match_positions);
        cudaFree(d_match_count);
        return 0;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(&h_match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_match_count > 0) {
        int* h_match_positions = (int*)malloc(h_match_count * sizeof(int));
        cudaMemcpy(h_match_positions, d_match_positions, 
                   h_match_count * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Store matches in chunk data with line information
        chunk_data->matches = (Match*)malloc(h_match_count * sizeof(Match));
        chunk_data->match_count = h_match_count;
        
        for (int i = 0; i < h_match_count; i++) {
            chunk_data->matches[i].position = h_match_positions[i];
            chunk_data->matches[i].thread_id = chunk_data->thread_id;
            
            // Extract the front part of the line for chromosome/line number info
            int line_pos = h_match_positions[i] - chunk_data->chunk_offset;
            int line_start = line_pos;
            
            // Find the start of the line
            while (line_start > 0 && chunk_data->text_chunk[line_start - 1] != '\n') {
                line_start--;
            }
            
            // Extract up to 199 characters from the start of the line (enough for chromo_lineNum|)
            int extract_len = 0;
            while (extract_len < 199 && 
                   (line_start + extract_len) < chunk_data->chunk_size &&
                   chunk_data->text_chunk[line_start + extract_len] != '\n' &&
                   chunk_data->text_chunk[line_start + extract_len] != '\0') {
                chunk_data->matches[i].line_info[extract_len] = chunk_data->text_chunk[line_start + extract_len];
                extract_len++;
                
                // Stop after we find the pipe character since we only need chromo_lineNum|
                if (chunk_data->text_chunk[line_start + extract_len - 1] == '|') {
                    // Get a bit more for safety
                    int extra = 0;
                    while (extra < 20 && 
                           (line_start + extract_len + extra) < chunk_data->chunk_size &&
                           chunk_data->text_chunk[line_start + extract_len + extra] != '\n' &&
                           chunk_data->text_chunk[line_start + extract_len + extra] != '\0') {
                        chunk_data->matches[i].line_info[extract_len + extra] = chunk_data->text_chunk[line_start + extract_len + extra];
                        extra++;
                    }
                    extract_len += extra;
                    break;
                }
            }
            chunk_data->matches[i].line_info[extract_len] = '\0';
        }
        
        free(h_match_positions);
    } else {
        chunk_data->matches = NULL;
        chunk_data->match_count = 0;
    }
    
    // Free device memory
    cudaFree(d_text);
    cudaFree(d_pattern);
    cudaFree(d_match_positions);
    cudaFree(d_match_count);
    
    return h_match_count;
}

// Function to extract chromosome and line number from a line and print result
void printMatchResult(const char* line, const char* pattern) {
    char chromo[100], lineNum[100];
    
    // Find the position of '_' and '|'
    const char *underscore = strchr(line, '_');
    const char *pipe = strchr(line, '|');
    
    if (underscore && pipe && underscore < pipe) {
        // Copy chromo (from beginning to underscore)
        size_t len_chromo = underscore - line;
        strncpy(chromo, line, len_chromo);
        chromo[len_chromo] = '\0';
        
        // Copy lineNum (from underscore+1 to pipe)
        size_t len_lineNum = pipe - underscore - 1;
        strncpy(lineNum, underscore + 1, len_lineNum);
        lineNum[len_lineNum] = '\0';
        
        printf("Pattern found at chromosome %s, at line %s\n", chromo, lineNum);
    }
}

int main() {
    char text_filename[] = "/home/cj/HPC_data/Human_genome_preprocessed.fna";
    char input_pattern[MAX_PATTERN_LENGTH];
    char* pattern = NULL;
    long file_size = 0;
    int pattern_length = 0;
    int num_threads = 0;
    
    // Get input from user
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
    
    // Get file size without loading entire file
    file_size = getFileSize(text_filename);
    if (file_size <= 0) {
        printf("Error: Cannot open or read file %s\n", text_filename);
        return 1;
    }
    
    // Use the pattern entered by user
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
    printf("- Chunk size: %d bytes\n", CHUNK_SIZE);
    
    // Check CUDA device
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
    
    // Calculate number of chunks based on file size
    int num_chunks = (file_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    printf("- Number of chunks: %d\n", num_chunks);
    printf("\nStarting hybrid search...\n");
    
    // Open file for streaming
    FILE* file = fopen(text_filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", text_filename);
        free(pattern);
        return 1;
    }
    
    // Allocate arrays to store results from all chunks
    int total_matches = 0;
    Match* all_matches = NULL;
    
    double start_time = omp_get_wtime();
    
    // Process file in chunks sequentially but use OpenMP+CUDA for each chunk
    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        int actual_chunk_size;
        char* text_chunk = readTextChunk(file, CHUNK_SIZE, &actual_chunk_size);
        
        if (!text_chunk || actual_chunk_size == 0) {
            break; // End of file or error
        }
        
        
        // Prepare chunk data for parallel processing
        ChunkData chunk_data;
        chunk_data.text_chunk = text_chunk;
        chunk_data.chunk_size = actual_chunk_size;
        chunk_data.chunk_offset = chunk_id * CHUNK_SIZE;
        chunk_data.thread_id = 0;
        chunk_data.matches = NULL;
        chunk_data.match_count = 0;
        
        // Process this chunk with CUDA
        int chunk_matches = processChunkWithCUDA(&chunk_data, pattern, pattern_length);
        
        // Collect results from this chunk
        if (chunk_matches > 0) {
            // Reallocate the global matches array
            all_matches = (Match*)realloc(all_matches, (total_matches + chunk_matches) * sizeof(Match));
            if (!all_matches) {
                printf("Error: Cannot allocate memory for matches\n");
                free(text_chunk);
                free(chunk_data.matches);
                break;
            }
            
            // Copy matches from this chunk
            for (int i = 0; i < chunk_matches; i++) {
                all_matches[total_matches + i] = chunk_data.matches[i];
            }
            total_matches += chunk_matches;
            
            free(chunk_data.matches);
        }
        
        free(text_chunk);
    }
    
    fclose(file);
    double end_time = omp_get_wtime();
    
    printf("\nSearch completed in %.6f seconds\n", end_time - start_time);
    
    // Process and print results
    if (total_matches > 0) {
        printf("============================================================================\n");
        
        // Sort matches by position first
        for (int i = 0; i < total_matches - 1; i++) {
            for (int j = 0; j < total_matches - i - 1; j++) {
                if (all_matches[j].position > all_matches[j + 1].position) {
                    Match temp = all_matches[j];
                    all_matches[j] = all_matches[j + 1];
                    all_matches[j + 1] = temp;
                }
            }
        }
        
        // Print results using the extracted line information
        for (int i = 0; i < total_matches; i++) {
            printMatchResult(all_matches[i].line_info, pattern);
        }
        
        printf("============================================================================\n");
        printf("Total matches found: %d\n", total_matches);
        printf("Time taken for hybrid OpenMP-CUDA search: %.6f seconds\n", end_time - start_time);
        
        free(all_matches);
    } else {
        printf("============================================================================\n");
        printf("Total matches found: 0\n");
        printf("Time taken for hybrid OpenMP-CUDA search: %.6f seconds\n", end_time - start_time);
    }
    
    // Cleanup
    free(pattern);
    
    return 0;
}
