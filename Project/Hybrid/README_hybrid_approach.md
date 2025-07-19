# Hybrid CUDA-OpenMP Search Implementation

## Overview
This implementation demonstrates a **true hybrid approach** where OpenMP threads coordinate CUDA operations on a single GPU device, rather than distributing work across multiple GPUs.

## Architecture

### Two-Level Parallelization Strategy

#### Level 1: OpenMP Thread Coordination
- **Data Preprocessing**: OpenMP threads work in parallel to extract metadata (chromosome and line numbers) from input lines
- **Result Processing**: OpenMP threads process search results in parallel
- **Thread Synchronization**: Uses barriers to coordinate between preprocessing, CUDA execution, and result processing phases

#### Level 2: CUDA GPU Execution
- **Pattern Matching**: GPU performs massively parallel string matching using CUDA threads
- **Memory Management**: Asynchronous memory transfers using CUDA streams
- **Shared Memory**: CUDA kernel uses shared memory for pattern caching to improve performance

## Key Components

### 1. **Hybrid Processing Function**
```c
void processFileChunkWithOpenMP(GPUContext *ctx, FILE *file, FileChunk *chunk, 
                               const char *pattern, int numCpuThreads)
```

This function implements the core hybrid logic:

1. **Sequential File I/O**: File reading remains sequential (required for file operations)
2. **Parallel Metadata Extraction**: 
   ```c
   #pragma omp parallel num_threads(numCpuThreads)
   {
       // Each thread processes its assigned lines for metadata extraction
       for (int i = start_idx; i < end_idx; i++) {
           // Extract chromosome and line numbers
       }
   }
   ```
3. **Master Thread CUDA Operations**:
   ```c
   #pragma omp master
   {
       // Only master thread handles CUDA operations
       cudaMemcpyAsync(...);
       hybridSearchKernel<<<...>>>();
       cudaMemcpyAsync(...);
   }
   ```
4. **Parallel Result Processing**:
   ```c
   #pragma omp for schedule(dynamic, 4)
   for (int i = 0; i < matchCount; i++) {
       // Process each match in parallel
   }
   ```

### 2. **CUDA Kernel Optimization**
- **Shared Memory**: Pattern is loaded into shared memory for faster access
- **Thread Efficiency**: Each CUDA thread processes one line
- **Atomic Operations**: Thread-safe result counting using `atomicAdd`

### 3. **Memory Management**
- **Pinned Host Memory**: Uses `cudaMallocHost` for better transfer performance
- **Asynchronous Transfers**: Overlaps computation and data transfer
- **Multiple Streams**: Uses 4 CUDA streams for better GPU utilization

## Benefits of This Approach

### 1. **Optimal Resource Utilization**
- **CPU Cores**: OpenMP threads keep CPU cores busy during metadata processing
- **GPU Cores**: CUDA threads handle compute-intensive pattern matching
- **Memory Bandwidth**: Asynchronous transfers hide memory latency

### 2. **Scalability**
- **Thread Count**: Adjustable OpenMP thread count based on CPU cores
- **Batch Size**: Configurable batch processing for memory efficiency
- **Stream Count**: Multiple CUDA streams for better GPU utilization

### 3. **Load Balancing**
- **Dynamic Scheduling**: OpenMP uses dynamic scheduling for result processing
- **Work Distribution**: Metadata extraction is evenly distributed among threads
- **Synchronization**: Proper barriers ensure all threads coordinate effectively

## Performance Characteristics

### Expected Speedup Sources
1. **Parallel Metadata Extraction**: N-way speedup where N = number of OpenMP threads
2. **GPU Pattern Matching**: Massive parallelism for string searching
3. **Parallel Result Processing**: Distributed output formatting
4. **Overlapped Execution**: CPU and GPU work simultaneously on different tasks

### Bottlenecks
1. **File I/O**: Sequential file reading limits scalability
2. **Memory Transfers**: Host-device communication overhead
3. **Synchronization**: Barrier overhead between OpenMP threads

## Usage

```bash
# Compile
nvcc -o hybridSearch hybridSearch.cu -Xcompiler -fopenmp

# Run
./hybridSearch
```

**Input Parameters:**
- Pattern to search for
- Number of OpenMP threads (1 to max available CPU cores)

**Output:**
- Match results with chromosome and line information
- Performance metrics including GPU time and wall clock time
- Efficiency ratio showing GPU utilization

## Configuration Options

### Compile-time Constants
- `BATCH_SIZE`: Number of lines processed per batch (65536)
- `THREADS_PER_BLOCK`: CUDA threads per block (512)
- `NUM_STREAMS`: Number of CUDA streams (4)
- `MAX_LINE_LENGTH`: Maximum characters per line (105)

### Runtime Parameters
- **OpenMP Threads**: Adjustable based on CPU core count
- **Pattern**: Any string pattern to search for
- **File Path**: Configurable input file location

## Technical Implementation Details

### OpenMP Synchronization Points
1. **After Metadata Extraction**: All threads complete preprocessing
2. **After CUDA Operations**: Master thread completes GPU work
3. **During Result Processing**: Dynamic work distribution

### CUDA Memory Pattern
1. **Host → Device**: Asynchronous transfer of line data
2. **Kernel Execution**: Parallel pattern matching
3. **Device → Host**: Asynchronous transfer of results

### Error Handling
- GPU context initialization validation
- CUDA operation error checking
- Memory allocation failure handling
- File I/O error management

This implementation provides a true hybrid computing solution that effectively combines the strengths of both OpenMP (CPU parallelization) and CUDA (GPU acceleration) in a coordinated manner.
