#!/bin/bash

# Build and Run Script for Hybrid CUDA-OpenMP Search System

echo "========================================================================"
echo "Hybrid CUDA-OpenMP Search System - Build & Run"
echo "========================================================================"

# Check prerequisites
echo "Checking prerequisites..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ NVCC (CUDA compiler) not found. Please install CUDA toolkit."
    echo "   Ubuntu/Debian: sudo apt-get install nvidia-cuda-toolkit"
    exit 1
fi

# Check OpenMP support
if ! gcc -fopenmp -x c /dev/null -o /tmp/test_omp 2>/dev/null; then
    echo "❌ OpenMP not supported. Please install OpenMP development libraries."
    echo "   Ubuntu/Debian: sudo apt-get install libomp-dev"
    exit 1
fi

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "⚠️  GPU not detected or NVIDIA drivers not installed."
    echo "   The program will still compile but may not run efficiently."
fi

echo "✅ Prerequisites check completed"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
make clean > /dev/null 2>&1

# Build the project
echo "Building hybrid search system..."
if make all; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "========================================================================"
echo "System Information"
echo "========================================================================"

# Show system info
make info

echo ""
echo "========================================================================"
echo "Running Hybrid Search System"
echo "========================================================================"

# Check if input file exists
INPUT_FILE="/home/cj/HPC_data/Human_genome_preprocessed.fna"
if [ ! -f "$INPUT_FILE" ]; then
    echo "⚠️  Input file not found: $INPUT_FILE"
    echo ""
    echo "To test the system, you need a preprocessed genomic data file."
    echo "Expected format: Each line should contain genomic sequence data"
    echo "with metadata in the format: chrX_linenum|sequence_data"
    echo ""
    echo "You can create a test file by running:"
    echo "  echo 'chr1_1|ATCGATCGATCGATCG' > test_input.fna"
    echo "  echo 'chr1_2|GGCCGGCCGGCCGGCC' >> test_input.fna"
    echo "  echo 'chr2_1|TACGTACGTACGTACG' >> test_input.fna"
    echo ""
    echo "Then modify the input file path in hybridSearch.cu"
    exit 1
fi

echo "Input file found: $INPUT_FILE"
echo "File size: $(du -h "$INPUT_FILE" | cut -f1)"
echo "Number of lines: $(wc -l < "$INPUT_FILE")"
echo ""

# Run the program
echo "Starting hybrid search..."
echo "Enter a DNA pattern to search for (e.g., ATCG, GGCC, TACG)"
echo ""

./hybridSearch

echo ""
echo "========================================================================"
echo "Execution completed!"
echo "========================================================================"

# Offer to run benchmark
echo ""
read -p "Would you like to run performance benchmarks? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running performance benchmarks..."
    ./benchmark.sh
fi

echo ""
echo "Thank you for using the Hybrid CUDA-OpenMP Search System!"
