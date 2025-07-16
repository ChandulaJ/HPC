# HPC Genome Search Frontend

A modern web-based frontend for running High-Performance Computing genome search implementations including Serial, OpenMP, and CUDA versions.

## Features

- **🧬 Data Preprocessing**: Prepare genome data with chromosome and line number formatting
- **🔄 Serial Search**: Single-threaded pattern matching
- **⚡ OpenMP Parallel Search**: Multi-threaded CPU parallelization with configurable thread count
- **🚀 CUDA GPU Search**: GPU-accelerated pattern matching using NVIDIA CUDA
- **📊 Real-time Output**: Live execution results and performance metrics
- **🎯 Interactive UI**: Modern, responsive web interface
- **⏱️ Performance Tracking**: Execution time and match count reporting

## Prerequisites

### Required
- **Node.js** (v14 or higher) - [Download](https://nodejs.org/)
- **npm** (comes with Node.js)
- **GCC compiler** - For serial and OpenMP implementations
  ```bash
  sudo apt-get install gcc
  ```

### Optional
- **NVIDIA CUDA Toolkit** - For GPU acceleration
  ```bash
  # Ubuntu/Debian
  sudo apt-get install nvidia-cuda-toolkit
  ```

### Data Requirements
- Genome data file: `/home/cj/HPC_data/Human_genome.fna`
- This will be processed to: `/home/cj/HPC_data/Human_genome_preprocessed.fna`

## Quick Start

1. **Clone and Navigate**
   ```bash
   cd /home/cj/HPC/frontend
   ```

2. **Auto Setup and Start**
   ```bash
   ./start.sh
   ```
   This script will:
   - Check system requirements
   - Install dependencies
   - Verify compilers
   - Start the server

3. **Manual Setup** (alternative)
   ```bash
   npm install
   npm start
   ```

4. **Access the Interface**
   Open your browser and go to: `http://localhost:3000`

## Usage Guide

### 1. Data Preprocessing
- Click **"Run Preprocessing"** first to prepare the genome data
- This adds chromosome identifiers and line numbers to the raw genome file
- Required before running any search operations

### 2. Serial Search
- Enter a DNA pattern (e.g., "ATCG", "GCTA")
- Click **"Run Serial Search"**
- Uses single-threaded CPU processing

### 3. OpenMP Parallel Search
- Enter a DNA pattern
- Specify number of threads (1-32, default: 4)
- Click **"Run OpenMP Search"**
- Uses multi-threaded CPU parallelization

### 4. CUDA GPU Search
- Enter a DNA pattern
- Click **"Run CUDA Search"**
- Uses GPU acceleration (requires NVIDIA GPU with CUDA support)

### Output Information
Each execution provides:
- **Match Results**: Chromosome and line locations of pattern matches
- **Total Matches**: Count of all pattern occurrences
- **Execution Time**: Performance timing in seconds
- **Real-time Output**: Live progress and status updates

## API Endpoints

The backend provides REST API endpoints:

- `POST /api/preprocessing` - Run data preprocessing
- `POST /api/serial` - Run serial search
  ```json
  { "pattern": "ATCG" }
  ```
- `POST /api/openmp` - Run OpenMP search
  ```json
  { "pattern": "ATCG", "threads": 4 }
  ```
- `POST /api/cuda` - Run CUDA search
  ```json
  { "pattern": "ATCG" }
  ```
- `GET /api/health` - Health check
- `GET /api/system` - System information

## Project Structure

```
/home/cj/HPC/
├── frontend/
│   ├── index.html          # Web interface
│   ├── server.js           # Node.js backend
│   ├── package.json        # Dependencies
│   ├── start.sh           # Setup script
│   └── README.md          # This file
└── Project/
    ├── Preprocessing/      # Data preprocessing
    ├── Serial/            # Serial implementation
    ├── OpenMp/           # OpenMP implementation
    └── CUDA/             # CUDA implementation
```

## Performance Comparison

The interface allows you to easily compare performance between different implementations:

1. **Serial**: Baseline single-threaded performance
2. **OpenMP**: Scalable multi-threaded CPU performance
3. **CUDA**: Massive parallel GPU acceleration

Example results will show execution times and speedup factors.

## Troubleshooting

### Common Issues

1. **"CUDA search failed"**
   - Ensure NVIDIA GPU with CUDA support is available
   - Install CUDA toolkit: `sudo apt-get install nvidia-cuda-toolkit`
   - Verify with: `nvcc --version`

2. **"OpenMP search failed"**
   - Install GCC: `sudo apt-get install gcc`
   - Verify OpenMP support: `gcc -fopenmp --version`

3. **"File not found" errors**
   - Ensure genome data file exists: `/home/cj/HPC_data/Human_genome.fna`
   - Run preprocessing first to create processed version

4. **Port already in use**
   - Change port in server.js or kill existing process:
   ```bash
   sudo lsof -t -i:3000 | xargs kill -9
   ```

### System Requirements Check

```bash
# Check Node.js
node --version

# Check npm
npm --version

# Check GCC
gcc --version

# Check CUDA (optional)
nvcc --version

# Check genome data
ls -la /home/cj/HPC_data/
```

## Development

### Running in Development Mode
```bash
npm run dev  # Uses nodemon for auto-restart
```

### Customization
- Modify `index.html` for UI changes
- Update `server.js` for backend logic
- Adjust paths in server.js if project structure differs

## License

MIT License - See project root for details.

## Support

For issues related to:
- **Frontend/Backend**: Check server logs and browser console
- **Compilation errors**: Verify GCC/CUDA installation
- **Performance**: Monitor system resources during execution

---

**Happy Genome Searching! 🧬🚀**
