
# Genome Pattern Search Accelerator

A high-performance computing system for searching DNA patterns in large genomic datasets using various parallel computing approaches.

## 🧬 Overview

This project provides a comprehensive solution for efficiently searching DNA patterns within large genome FASTA files. It implements and compares multiple pattern matching algorithms across different computing models:

- **Serial Implementation**: Single-threaded baseline approach
- **OpenMP Implementation**: Multi-threaded CPU parallelization
- **CUDA Implementation**: GPU-accelerated pattern matching
- **Hybrid Implementation**: Combined CPU-GPU approach using both OpenMP and CUDA

The system includes a full preprocessing pipeline, multiple search algorithm variants, and a modern web-based user interface for ease of use.

## 📋 Project Structure

```
├── frontend/              # Web interface for the search system
│   ├── index.html         # Main user interface
│   ├── server.js          # Express.js backend server
│   ├── package.json       # Node.js dependencies
│   └── start.sh           # Quick start script
├── Project/
│   ├── CUDA/              # GPU-accelerated implementation
│   │   ├── cudaSearch.cu  # CUDA kernel implementation
│   ├── Hybrid/            # Combined CPU-GPU implementation
│   │   ├── hybridSearch.cu# OpenMP + CUDA implementation
│   │   └── Makefile       # Build configuration
│   ├── OpenMp/            # Multi-threaded CPU implementation
│   │   ├── openMPsearch.c # OpenMP parallel implementation
│   ├── Preprocessing/     # Data preparation utilities
│   │   ├── addLineNumbers.c# Line indexing for chromosome data
│   └── Serial/            # Single-threaded implementations
│       ├── serialSearch.c # Basic pattern matching
│       ├── KMPserialSearch.c # KMP algorithm implementation
│       └── serialSearchBMAlgo.c # Boyer-Moore algorithm
```

## ⚙️ Features

- **Multiple Search Algorithms**:
  - Simple string matching (naive algorithm)
  - Knuth-Morris-Pratt (KMP) algorithm
  - Boyer-Moore algorithm
  - Parallel variants of each

- **High-Performance Computing**:
  - CPU parallelization with OpenMP
  - GPU acceleration with CUDA
  - Hybrid computing model combining both approaches

- **System Optimizations**:
  - Shared memory utilization in CUDA
  - CUDA streams for overlapped computation
  - Memory coalescing for better throughput
  - Batch processing of genomic data
  - Pinned memory for faster CPU-GPU transfers

- **Data Management**:
  - Chromosome-wise indexing
  - Line number tracking for accurate match reporting
  - Efficient FASTA file parsing

- **User Interface**:
  - Web-based frontend for running all implementations
  - Real-time output and performance metrics
  - Interactive workflow for genomic research

## 📊 Algorithm Comparison

| Algorithm | Implementation | Parallelization | Best Use Case |
|-----------|---------------|----------------|--------------|
| Naive Search | serialSearch.c | None | Small patterns, baseline comparison |
| KMP | KMPserialSearch.c | None | Patterns with many repeating characters |
| Boyer-Moore | serialSearchBMAlgo.c | None | Long patterns with varied characters |
| OpenMP | openMPsearch.c | Multi-threaded CPU | Mid-sized datasets, CPU-only systems |
| CUDA | cudaSearch.cu | GPU | Very large datasets with GPU access |
| Hybrid | hybridSearch.cu | CPU + GPU | Maximum performance on heterogeneous systems |

## 🚀 Getting Started

### Prerequisites

- **C/C++ Compiler** (GCC 9+ recommended)
- **CUDA Toolkit** (11.0+ for GPU acceleration)
- **OpenMP** (included with most C compilers)
- **Node.js** (v14+ for the web frontend)
- **Genomic data** in FASTA format (preprocessed for line indexing)

### Building the Project

#### Serial and OpenMP Implementations

```bash
# Build Serial implementations
cd Project/Serial
gcc serialSearch.c -o serialSearch
gcc KMPserialSearch.c -o KMPserialSearch
gcc serialSearchBMAlgo.c -o serialSearchBMAlgo

# Build OpenMP implementation
cd ../OpenMp
gcc -fopenmp openMPsearch.c -o openMPsearch
```

#### CUDA Implementation

```bash
cd ../CUDA
nvcc -O3 cudaSearch.cu -o cudaSearch
```

#### Hybrid Implementation

```bash
cd ../Hybrid
make
```

### Data Preprocessing

Before searching, the genomic data must be preprocessed to add line numbers and chromosome identifiers:

```bash
cd ../Preprocessing
gcc addLineNumbers.c -o addLineNumbers
./addLineNumbers
```

Input your genomic FASTA file path when prompted (default: `/home/cj/HPC_data/Human_genome.fna`).

## 💻 Web Interface

The project includes a modern web interface for running all implementations and comparing their performance:

### Starting the Frontend

```bash
cd ../../frontend
./start.sh
```

This will install dependencies and start the web server. Access the UI at `http://localhost:3000`.

### Features of the Web Interface

- Run preprocessing on genomic data
- Execute any of the search implementations
- Configure parameters (pattern, thread count, etc.)
- View real-time search results
- Compare execution times across implementations
- Access detailed match information

## 🔬 Implementation Details

### Serial Search Algorithms

The project includes three serial algorithm implementations:

1. **Basic Pattern Search** (serialSearch.c):
   - Simple character-by-character comparison
   - O(m*n) time complexity where m is pattern length and n is text length
   
2. **Knuth-Morris-Pratt** (KMPserialSearch.c):
   - Efficient string matching using pattern preprocessing
   - O(m+n) time complexity
   - Avoids redundant comparisons by using pattern information
   
3. **Boyer-Moore** (serialSearchBMAlgo.c):
   - Right-to-left scanning with efficient skip values
   - Can skip portions of the input text
   - Good for large alphabets like DNA sequences

### OpenMP Implementation

The OpenMP implementation (openMPsearch.c) parallelizes the search across multiple CPU threads:

- Divides input data into chunks processed by separate threads
- Uses thread-safe result collection
- Configurable thread count for different CPU architectures
- Maintains result order for accurate reporting

### CUDA Implementation

The CUDA implementation (cudaSearch.cu) utilizes GPU acceleration:

- Thousands of GPU threads search in parallel
- Shared memory optimization for pattern storage
- Memory coalescing for efficient GPU memory access
- Batch processing to handle large datasets
- Asynchronous operations with CUDA streams

### Hybrid Implementation

The hybrid approach (hybridSearch.cu) combines both OpenMP and CUDA:

- OpenMP handles file I/O and preprocessing
- CUDA performs pattern matching on GPU
- Dynamic workload distribution between CPU and GPU
- Overlapped computation and data transfer
- Pinned memory for zero-copy memory transfers

## 📈 Performance Comparison

Typical performance comparisons on large genomic datasets (example for 100MB input):

| Implementation | Threads/Cores | Execution Time | Speedup |
|----------------|--------------|---------------|---------|
| Serial (naive) | 1 | 12.5s | 1x |
| KMP Serial | 1 | 8.2s | 1.5x |
| Boyer-Moore | 1 | 6.7s | 1.9x |
| OpenMP | 8 | 2.1s | 6x |
| CUDA | 2048 | 0.9s | 14x |
| Hybrid | 8 CPU + GPU | 0.5s | 25x |

*Note: Actual performance varies based on hardware configuration, pattern complexity, and dataset size.*

## 🌐 API Reference

The frontend server provides the following API endpoints:

- `POST /api/preprocessing` - Run data preprocessing
- `POST /api/serial` - Run serial search
  ```json
  { "pattern": "ATCG" }
  ```
- `POST /api/kmp` - Run KMP algorithm
  ```json
  { "pattern": "ATCG" }
  ```
- `POST /api/bm` - Run Boyer-Moore algorithm
  ```json
  { "pattern": "ATCG" }
  ```
- `POST /api/openmp` - Run OpenMP parallel search
  ```json
  { "pattern": "ATCG", "threads": 4 }
  ```
- `POST /api/cuda` - Run CUDA GPU search
  ```json
  { "pattern": "ATCG" }
  ```
- `POST /api/hybrid` - Run Hybrid CPU-GPU search
  ```json
  { "pattern": "ATCG" }
  ```

## ⚠️ Requirements and Limitations

- **Hardware Requirements**:
  - CPU: Multi-core processor (4+ cores recommended)
  - RAM: 8GB+ (16GB+ recommended for large genomes)
  - GPU: NVIDIA GPU with CUDA support (Compute Capability 5.0+)
  - Storage: Fast SSD for large genomic datasets

- **Software Requirements**:
  - Linux-based OS (Ubuntu 20.04+ recommended)
  - CUDA Toolkit 11.0+
  - GCC 9+ with OpenMP support
  - Node.js 14+ for frontend

- **Dataset Requirements**:
  - Standard FASTA format (`.fna` or `.fa`)
  - Preprocessed with chromosome identifiers and line numbers

## 📜 License

This project is available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Areas for improvement include:

- Additional search algorithms
- Enhanced visualization of search results
- Support for more genomic file formats
- Optimizations for different hardware architectures
- Cloud deployment options

## 👥 Authors

- Chandula Jayasundara - Initial development and optimization

## 🙏 Acknowledgments

- Bioinformatics research community for genomic data standards
- NVIDIA for CUDA tooling and documentation
- OpenMP Architecture Review Board


