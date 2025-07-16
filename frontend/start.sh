#!/bin/bash

# HPC Genome Search Frontend Setup and Startup Script

echo "🧬 HPC Genome Search Frontend Setup"
echo "=================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    echo "Visit: https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ npm version: $(npm --version)"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check for CUDA (optional)
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA compiler found: $(nvcc --version | head -n 1)"
else
    echo "⚠️  CUDA compiler not found - CUDA searches will not work"
fi

# Check for GCC (required for OpenMP and Serial)
if command -v gcc &> /dev/null; then
    echo "✅ GCC compiler found: $(gcc --version | head -n 1)"
else
    echo "❌ GCC compiler not found - Serial and OpenMP searches will not work"
fi

# Check if genome data file exists
GENOME_FILE="/home/cj/HPC_data/Human_genome.fna"
if [ -f "$GENOME_FILE" ]; then
    echo "✅ Genome data file found: $GENOME_FILE"
else
    echo "⚠️  Genome data file not found: $GENOME_FILE"
    echo "   Make sure the genome data is available for processing"
fi

# Create executables directory if it doesn't exist
mkdir -p ../Project/Preprocessing ../Project/Serial ../Project/OpenMp ../Project/CUDA

echo ""
echo "🚀 Starting HPC Genome Search Frontend Server..."
echo "   Server will be available at: http://localhost:3000"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the server
npm start
