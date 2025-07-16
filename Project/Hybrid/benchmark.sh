#!/bin/bash

# Performance Comparison Script for Hybrid CUDA-OpenMP Search System
# This script compares performance between different implementations

echo "========================================================================"
echo "Performance Comparison: Hybrid vs CUDA vs OpenMP vs Serial"
echo "========================================================================"

# Configuration
TEST_PATTERNS=("ATCG" "GGCC" "TACG" "AAAAAA" "CGCGCG")
NUM_CPU_THREADS=$(nproc)
RESULTS_FILE="performance_results.txt"

echo "System Information:"
echo "- CPU Cores: $NUM_CPU_THREADS"
echo "- GPU: $(nvidia-smi -L 2>/dev/null | head -1 | cut -d':' -f2- || echo 'Not detected')"
echo "- Date: $(date)"
echo "- Input file: /home/cj/HPC_data/Human_genome_preprocessed.fna"
echo ""

# Check if all executables exist
echo "Checking available implementations..."
IMPLEMENTATIONS=()

if [ -f "../Serial/serialSearch" ]; then
    IMPLEMENTATIONS+=("Serial")
    echo "✓ Serial implementation found"
else
    echo "✗ Serial implementation not found"
fi

if [ -f "../OpenMp/openMPsearch" ]; then
    IMPLEMENTATIONS+=("OpenMP")
    echo "✓ OpenMP implementation found"
else
    echo "✗ OpenMP implementation not found"
fi

if [ -f "../CUDA/cudaSearch" ]; then
    IMPLEMENTATIONS+=("CUDA")
    echo "✓ CUDA implementation found"
else
    echo "✗ CUDA implementation not found"
fi

if [ -f "./hybridSearch" ]; then
    IMPLEMENTATIONS+=("Hybrid")
    echo "✓ Hybrid implementation found"
else
    echo "✗ Hybrid implementation not found - building now..."
    make all
    if [ -f "./hybridSearch" ]; then
        IMPLEMENTATIONS+=("Hybrid")
        echo "✓ Hybrid implementation built successfully"
    else
        echo "✗ Failed to build hybrid implementation"
    fi
fi

echo ""

# Initialize results file
echo "Performance Comparison Results - $(date)" > "$RESULTS_FILE"
echo "================================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Function to run performance test
run_performance_test() {
    local impl=$1
    local pattern=$2
    local executable=$3
    local input_method=$4
    
    echo "Testing $impl with pattern '$pattern'..."
    
    start_time=$(date +%s.%N)
    
    case $input_method in
        "auto_input")
            # For implementations that take automatic input
            echo "$pattern" | timeout 300 "$executable" > temp_output.txt 2>&1
            ;;
        "openmp_input")
            # For OpenMP that needs thread count
            echo -e "$pattern\n$NUM_CPU_THREADS" | timeout 300 "$executable" > temp_output.txt 2>&1
            ;;
        "hybrid_input")
            # For Hybrid that needs thread count
            echo -e "$pattern\n$NUM_CPU_THREADS" | timeout 300 "$executable" > temp_output.txt 2>&1
            ;;
    esac
    
    exit_code=$?
    end_time=$(date +%s.%N)
    
    if [ $exit_code -eq 0 ]; then
        # Extract timing and match information from output
        matches=$(grep -o "Total matches found: [0-9]*" temp_output.txt | grep -o "[0-9]*" | tail -1)
        time_taken=$(grep -o "Time taken.*: [0-9.]*" temp_output.txt | grep -o "[0-9.]*" | tail -1)
        
        if [ -z "$time_taken" ]; then
            # Calculate elapsed time if not provided by program
            time_taken=$(echo "$end_time - $start_time" | bc)
        fi
        
        echo "  ✓ Completed in ${time_taken}s, found ${matches:-0} matches"
        
        # Log to results file
        echo "$impl,$pattern,$time_taken,${matches:-0}" >> "$RESULTS_FILE"
    else
        echo "  ✗ Failed or timed out"
        echo "$impl,$pattern,TIMEOUT,0" >> "$RESULTS_FILE"
    fi
    
    rm -f temp_output.txt
}

# Main performance testing loop
echo "Running performance tests..."
echo "Implementation,Pattern,Time(s),Matches" >> "$RESULTS_FILE"

for pattern in "${TEST_PATTERNS[@]}"; do
    echo ""
    echo "======== Testing Pattern: $pattern ========"
    
    for impl in "${IMPLEMENTATIONS[@]}"; do
        case $impl in
            "Serial")
                run_performance_test "$impl" "$pattern" "../Serial/serialSearch" "auto_input"
                ;;
            "OpenMP")
                run_performance_test "$impl" "$pattern" "../OpenMp/openMPsearch" "openmp_input"
                ;;
            "CUDA")
                run_performance_test "$impl" "$pattern" "../CUDA/cudaSearch" "auto_input"
                ;;
            "Hybrid")
                run_performance_test "$impl" "$pattern" "./hybridSearch" "hybrid_input"
                ;;
        esac
        sleep 1  # Brief pause between tests
    done
done

echo ""
echo "========================================================================"
echo "Performance Analysis"
echo "========================================================================"

# Generate performance summary
python3 << 'EOF'
import csv
import sys

try:
    # Read results
    results = {}
    with open('performance_results.txt', 'r') as f:
        lines = f.readlines()
        
    # Find CSV data start
    csv_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Implementation,"):
            csv_start = i
            break
    
    # Parse CSV data
    csv_data = lines[csv_start:]
    reader = csv.DictReader(csv_data)
    
    for row in reader:
        impl = row['Implementation']
        pattern = row['Pattern']
        time_str = row['Time(s)']
        matches = int(row['Matches'])
        
        if impl not in results:
            results[impl] = {}
        
        if time_str != 'TIMEOUT':
            time_val = float(time_str)
            results[impl][pattern] = {'time': time_val, 'matches': matches}
        else:
            results[impl][pattern] = {'time': float('inf'), 'matches': 0}
    
    # Calculate averages and speedups
    print("Summary by Implementation:")
    print("-" * 60)
    
    avg_times = {}
    for impl in results:
        times = [data['time'] for data in results[impl].values() if data['time'] != float('inf')]
        if times:
            avg_time = sum(times) / len(times)
            avg_times[impl] = avg_time
            print(f"{impl:12s}: {avg_time:8.3f}s average")
        else:
            print(f"{impl:12s}: TIMEOUT")
    
    print("\nSpeedup Comparison (vs Serial):")
    print("-" * 60)
    
    if 'Serial' in avg_times:
        baseline = avg_times['Serial']
        for impl, avg_time in avg_times.items():
            if impl != 'Serial':
                speedup = baseline / avg_time
                print(f"{impl:12s}: {speedup:6.2f}x faster than Serial")
    
    print("\nPattern-wise Performance (Time in seconds):")
    print("-" * 80)
    print(f"{'Pattern':<10s}", end='')
    for impl in sorted(results.keys()):
        print(f"{impl:>12s}", end='')
    print()
    
    patterns = set()
    for impl_data in results.values():
        patterns.update(impl_data.keys())
    
    for pattern in sorted(patterns):
        print(f"{pattern:<10s}", end='')
        for impl in sorted(results.keys()):
            if pattern in results[impl]:
                time_val = results[impl][pattern]['time']
                if time_val == float('inf'):
                    print(f"{'TIMEOUT':>12s}", end='')
                else:
                    print(f"{time_val:>12.3f}", end='')
            else:
                print(f"{'N/A':>12s}", end='')
        print()

except Exception as e:
    print(f"Error analyzing results: {e}")
    print("Raw results file content:")
    try:
        with open('performance_results.txt', 'r') as f:
            print(f.read())
    except:
        print("Could not read results file")
EOF

echo ""
echo "========================================================================"
echo "Detailed Results"
echo "========================================================================"
echo "Full results saved to: $RESULTS_FILE"
echo ""

# Show system resource usage
echo "System Resource Usage:"
echo "- CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "- GPU Usage: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
    echo "- GPU Memory: $(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits)%"
fi

echo ""
echo "Test completed at $(date)"
echo "Results saved to $RESULTS_FILE"
