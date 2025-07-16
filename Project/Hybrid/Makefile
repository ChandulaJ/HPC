# Makefile for Hybrid CUDA-OpenMP Search System
# Combines CUDA GPU acceleration with OpenMP CPU parallelization

# Compiler settings
NVCC = nvcc
CC = gcc

# Compilation flags
CUDA_FLAGS = -O3 -arch=sm_50 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75
OPENMP_FLAGS = -fopenmp
OPTIMIZATION_FLAGS = -O3
WARNING_FLAGS = -Wall -Wextra
CUDA_LIBS = -lcuda -lcudart

# Target executable
TARGET = hybridSearch
SOURCE = hybridSearch.cu

# Default target
all: $(TARGET)

# Build the hybrid search executable
$(TARGET): $(SOURCE)
	@echo "Building Hybrid CUDA-OpenMP Search System..."
	@echo "CUDA Architecture: SM 5.0+ (Maxwell, Pascal, Volta, Turing, Ampere)"
	@echo "OpenMP: Enabled for CPU parallelization"
	$(NVCC) $(CUDA_FLAGS) $(OPTIMIZATION_FLAGS) -Xcompiler "$(OPENMP_FLAGS) $(WARNING_FLAGS)" \
		$(SOURCE) -o $(TARGET) $(CUDA_LIBS)
	@echo "Build completed successfully!"
	@echo "Usage: ./$(TARGET)"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET) *.o
	@echo "Clean completed!"

# Install required dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential nvidia-cuda-toolkit libomp-dev
	@echo "Dependencies installed!"

# Test the build
test: $(TARGET)
	@echo "Testing hybrid search system..."
	@echo "Please ensure your input file exists at: /home/cj/HPC_data/Human_genome_preprocessed.fna"
	./$(TARGET)

# Performance benchmark
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	@echo "Testing with sample patterns..."
	@for pattern in "ATCG" "GGCC" "TACG"; do \
		echo "Testing pattern: $$pattern"; \
		echo "$$pattern" | ./$(TARGET); \
		echo ""; \
	done

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "NVCC found: $$(nvcc --version | grep release)"; \
	else \
		echo "ERROR: NVCC not found. Please install CUDA toolkit."; \
		exit 1; \
	fi
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "GPU Information:"; \
		nvidia-smi -L; \
	else \
		echo "WARNING: nvidia-smi not found. GPU may not be available."; \
	fi

# Check OpenMP support
check-openmp:
	@echo "Checking OpenMP support..."
	@echo '#include <omp.h>' > test_omp.c
	@echo 'int main() { return omp_get_max_threads(); }' >> test_omp.c
	@if $(CC) $(OPENMP_FLAGS) test_omp.c -o test_omp 2>/dev/null; then \
		echo "OpenMP support: OK"; \
		echo "Max threads available: $$(./test_omp)"; \
	else \
		echo "ERROR: OpenMP not supported or not installed."; \
	fi
	@rm -f test_omp.c test_omp

# Show system information
info: check-cuda check-openmp
	@echo ""
	@echo "=== System Information ==="
	@echo "CPU cores: $$(nproc)"
	@echo "Memory: $$(free -h | grep Mem | awk '{print $$2}')"
	@echo "Compiler versions:"
	@echo "  GCC: $$(gcc --version | head -1)"
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "  NVCC: $$(nvcc --version | grep release | awk '{print $$5 $$6}')"; \
	fi

# Help target
help:
	@echo "Hybrid CUDA-OpenMP Search System Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build the hybrid search executable (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  install-deps - Install required dependencies (Ubuntu/Debian)"
	@echo "  test         - Build and run the program"
	@echo "  benchmark    - Run performance benchmark with sample patterns"
	@echo "  check-cuda   - Check CUDA installation and GPU availability"
	@echo "  check-openmp - Check OpenMP support"
	@echo "  info         - Show system information"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - NVIDIA GPU with CUDA Compute Capability 5.0+"
	@echo "  - CUDA Toolkit 10.0+"
	@echo "  - GCC with OpenMP support"
	@echo "  - Input file: /home/cj/HPC_data/Human_genome_preprocessed.fna"

.PHONY: all clean install-deps test benchmark check-cuda check-openmp info help
