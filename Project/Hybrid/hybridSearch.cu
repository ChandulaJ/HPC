#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_PATTERN_LENGTH 1024
#define THREADS_PER_BLOCK 256
#define CHUNK_SIZE (1024 * 1024)
#define MAX_MATCHES_PER_CHUNK 10000

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < text_size) {
        int line_start = idx;
        while (line_start > 0 && text[line_start - 1] != '\n') {
            line_start--;
        }
        
        int line_end = idx;
        while (line_end < text_size && text[line_end] != '\n' && text[line_end] != '\0') {
            line_end++;
        }
        
        int line_length = line_end - line_start;
        
        if (idx == line_start && line_length >= pattern_length) {
            for (int i = 0; i <= line_length - pattern_length; i++) {
                bool match = true;
                
                for (int j = 0; j < pattern_length; j++) {
                    if (text[line_start + i + j] != pattern[j]) {
                        match = false;
                        break;
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

char* readTextChunk(FILE* file, int chunk_size, int* actual_size) {
    char* chunk = (char*)malloc(chunk_size + 1);
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
    
    cudaError_t err;
    
    err = cudaMalloc(&d_text, chunk_data->chunk_size);
    if (err != cudaSuccess) {
        printf("Error allocating device memory for text: %s\n", cudaGetErrorString(err));
        return 0;
    }
    
    err = cudaMalloc(&d_pattern, pattern_length);
    if (err != cudaSuccess) {
        printf("Error allocating device memory for pattern: %s\n", cudaGetErrorString(err));
        cleanupCudaMemory(d_text, NULL, NULL, NULL);
        return 0;
    }
    
    err = cudaMalloc(&d_match_positions, MAX_MATCHES_PER_CHUNK * sizeof(int));
    if (err != cudaSuccess) {
        printf("Error allocating device memory for match positions: %s\n", cudaGetErrorString(err));
        cleanupCudaMemory(d_text, d_pattern, NULL, NULL);
        return 0;
    }
    
    err = cudaMalloc(&d_match_count, sizeof(int));
    if (err != cudaSuccess) {
        printf("Error allocating device memory for match count: %s\n", cudaGetErrorString(err));
        cleanupCudaMemory(d_text, d_pattern, d_match_positions, NULL);
        return 0;
    }
    
    cudaMemcpy(d_text, chunk_data->text_chunk, chunk_data->chunk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern, pattern_length, cudaMemcpyHostToDevice);
    cudaMemset(d_match_count, 0, sizeof(int));
    
    int num_blocks = (chunk_data->chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    searchKernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_text, d_pattern, chunk_data->chunk_size, pattern_length,
        d_match_positions, d_match_count, chunk_data->chunk_offset);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        cleanupCudaMemory(d_text, d_pattern, d_match_positions, d_match_count);
        return 0;
    }
    
    cudaDeviceSynchronize();
    cudaMemcpy(&h_match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_match_count > 0) {
        int* h_match_positions = (int*)malloc(h_match_count * sizeof(int));
        cudaMemcpy(h_match_positions, d_match_positions, 
                   h_match_count * sizeof(int), cudaMemcpyDeviceToHost);
        
        chunk_data->matches = (Match*)malloc(h_match_count * sizeof(Match));
        chunk_data->match_count = h_match_count;
        
        for (int i = 0; i < h_match_count; i++) {
            chunk_data->matches[i].position = h_match_positions[i];
            
            int line_pos = h_match_positions[i] - chunk_data->chunk_offset;
            int line_start = line_pos;
            
            while (line_start > 0 && chunk_data->text_chunk[line_start - 1] != '\n') {
                line_start--;
            }
            
            int extract_len = 0;
            while (extract_len < 199 && 
                   (line_start + extract_len) < chunk_data->chunk_size &&
                   chunk_data->text_chunk[line_start + extract_len] != '\n' &&
                   chunk_data->text_chunk[line_start + extract_len] != '\0') {
                chunk_data->matches[i].line_info[extract_len] = chunk_data->text_chunk[line_start + extract_len];
                extract_len++;
                
                if (chunk_data->text_chunk[line_start + extract_len - 1] == '|') {
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
    
    cleanupCudaMemory(d_text, d_pattern, d_match_positions, d_match_count);
    return h_match_count;
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
    Match* all_matches = NULL;
    
    double start_time = omp_get_wtime();
    
    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        int actual_chunk_size;
        char* text_chunk = readTextChunk(file, CHUNK_SIZE, &actual_chunk_size);
        
        if (!text_chunk || actual_chunk_size == 0) break;
        
        ChunkData chunk_data = {text_chunk, actual_chunk_size, chunk_id * CHUNK_SIZE, NULL, 0};
        int chunk_matches = processChunkWithCUDA(&chunk_data, pattern, pattern_length);
        
        if (chunk_matches > 0) {
            all_matches = (Match*)realloc(all_matches, (total_matches + chunk_matches) * sizeof(Match));
            if (!all_matches) {
                printf("Error: Cannot allocate memory for matches\n");
                free(text_chunk);
                free(chunk_data.matches);
                break;
            }
            
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
    
    if (total_matches > 0) {
        printf("============================================================================\n");
        
        // Simple bubble sort for matches by position
        for (int i = 0; i < total_matches - 1; i++) {
            for (int j = 0; j < total_matches - i - 1; j++) {
                if (all_matches[j].position > all_matches[j + 1].position) {
                    Match temp = all_matches[j];
                    all_matches[j] = all_matches[j + 1];
                    all_matches[j + 1] = temp;
                }
            }
        }
        
        for (int i = 0; i < total_matches; i++) {
            printMatchResult(all_matches[i].line_info);
        }
        
        printf("============================================================================\n");
        free(all_matches);
    } else {
        printf("============================================================================\n");
    }
    
    printf("Total matches found: %d\n", total_matches);
    printf("Time taken for hybrid OpenMP-CUDA search: %.6f seconds\n", end_time - start_time);
    
    free(pattern);
    return 0;
}
