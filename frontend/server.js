const express = require('express');
const cors = require('cors');
const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('.')); // Serve static files from current directory

// Project paths
const PROJECT_ROOT = '/home/cj/HPC/Project';
const PREPROCESSING_PATH = path.join(PROJECT_ROOT, 'Preprocessing');
const SERIAL_PATH = path.join(PROJECT_ROOT, 'Serial');
const OPENMP_PATH = path.join(PROJECT_ROOT, 'OpenMp');
const CUDA_PATH = path.join(PROJECT_ROOT, 'CUDA');
const HYBRID_PATH = path.join(PROJECT_ROOT, 'Hybrid');

// Utility function to execute commands with timeout and better error handling
function executeCommand(command, args, options = {}) {
    return new Promise((resolve, reject) => {
        const startTime = Date.now();
        
        console.log(`Executing: ${command} ${args.join(' ')}`);
        console.log(`Working directory: ${options.cwd || process.cwd()}`);
        
        const child = spawn(command, args, {
            cwd: options.cwd || process.cwd(),
            stdio: ['pipe', 'pipe', 'pipe'],
            ...options
        });

        let stdout = '';
        let stderr = '';

        // Handle stdout
        child.stdout.on('data', (data) => {
            const text = data.toString();
            stdout += text;
            console.log('STDOUT:', text);
        });

        // Handle stderr
        child.stderr.on('data', (data) => {
            const text = data.toString();
            stderr += text;
            console.log('STDERR:', text);
        });

        // Handle process completion
        child.on('close', (code) => {
            const endTime = Date.now();
            const executionTime = ((endTime - startTime) / 1000).toFixed(3);
            
            console.log(`Process exited with code: ${code}`);
            console.log(`Total server time (including overhead): ${executionTime}s`);
            
            if (code === 0) {
                resolve({
                    success: true,
                    output: stdout,
                    error: stderr,
                    executionTime: parseFloat(executionTime),
                    exitCode: code
                });
            } else {
                reject({
                    success: false,
                    output: stdout,
                    error: stderr || `Process exited with code ${code}`,
                    executionTime: parseFloat(executionTime),
                    exitCode: code
                });
            }
        });

        // Handle process errors
        child.on('error', (error) => {
            console.error('Process error:', error);
            reject({
                success: false,
                error: `Failed to start process: ${error.message}`,
                executionTime: 0,
                exitCode: -1
            });
        });

        // Set timeout for long-running processes
        const timeout = options.timeout || 300000; // 5 minutes default
        setTimeout(() => {
            child.kill('SIGKILL');
            reject({
                success: false,
                error: 'Process timed out',
                executionTime: timeout / 1000,
                exitCode: -1
            });
        }, timeout);

        // Send input if provided
        if (options.input) {
            child.stdin.write(options.input);
            child.stdin.end();
        }
    });
}

// Parse execution results
function parseResults(output) {
    const results = {
        totalMatches: 0,
        executionTime: 0,
        details: [],
        processingRate: null,
        chunksProcessed: null
    };

    // Extract total matches
    const matchRegex = /Total matches found:\s*(\d+)/i;
    const matchMatch = output.match(matchRegex);
    if (matchMatch) {
        results.totalMatches = parseInt(matchMatch[1]);
    }

    // Extract execution time (prioritize program-specific timing)
    const timeRegexes = [
        /Time taken for (?:hybrid OpenMP-CUDA) search:\s*([\d.]+)\s*seconds/i,
        /Time taken for (?:serial|openmp|cuda|hybrid) (?:parallel\s*)?(?:OpenMP-CUDA\s*)?search:\s*([\d.]+)\s*seconds/i,
        /Time taken for (?:serial|openmp|cuda|hybrid).*?:\s*([\d.]+)\s*seconds/i,
        /Execution time:\s*([\d.]+)\s*seconds/i
    ];
    
    for (const regex of timeRegexes) {
        const timeMatch = output.match(regex);
        if (timeMatch) {
            results.executionTime = parseFloat(timeMatch[1]);
            break;
        }
    }

    // Extract processing rate (hybrid-specific)
    const rateMatch = output.match(/Processing rate:\s*([\d.]+)\s*MB\/s/i);
    if (rateMatch) {
        results.processingRate = parseFloat(rateMatch[1]);
    }

    // Extract chunks processed (hybrid-specific)
    const chunksMatch = output.match(/Chunks processed in parallel:\s*(\d+)/i);
    if (chunksMatch) {
        results.chunksProcessed = parseInt(chunksMatch[1]);
    }

    // Extract pattern matches
    const patternRegex = /Pattern found at chromosome\s+([^,]+),\s*at line\s+(\S+)/g;
    let match;
    while ((match = patternRegex.exec(output)) !== null) {
        results.details.push({
            chromosome: match[1].trim(),
            line: match[2].trim()
        });
    }

    return results;
}

// API Routes

// Preprocessing endpoint
app.post('/api/preprocessing', async (req, res) => {
    try {
        const { inputFilePath, outputFilePath } = req.body;
        
        if (!inputFilePath) {
            return res.status(400).json({
                success: false,
                error: 'Input file path is required'
            });
        }

        if (!outputFilePath) {
            return res.status(400).json({
                success: false,
                error: 'Output file path is required'
            });
        }

        console.log(`Starting preprocessing...`);
        console.log(`Input file: ${inputFilePath}`);
        console.log(`Output file: ${outputFilePath}`);
        
        // Check if executable exists, compile if needed
        const executablePath = path.join(PREPROCESSING_PATH, 'addLineNumbers');
        if (!fs.existsSync(executablePath)) {
            console.log('Compiling preprocessing program...');
            await executeCommand('gcc', ['-o', 'addLineNumbers', 'addLineNumbers.c'], {
                cwd: PREPROCESSING_PATH
            });
        }

        // Run preprocessing with input and output file paths
        const result = await executeCommand('./addLineNumbers', [], {
            cwd: PREPROCESSING_PATH,
            input: `${inputFilePath}\n${outputFilePath}\n`,
            timeout: 600000 // 10 minutes for large files
        });

        res.json({
            success: true,
            message: 'Preprocessing completed successfully',
            output: result.output,
            executionTime: result.executionTime,
            inputFilePath: inputFilePath,
            outputFilePath: outputFilePath
        });

    } catch (error) {
        console.error('Preprocessing error:', error);
        res.status(500).json({
            success: false,
            error: 'Preprocessing failed',
            details: error.error || error.message,
            output: error.output || ''
        });
    }
});

// Serial search endpoint
app.post('/api/serial', async (req, res) => {
    try {
        const { pattern } = req.body;
        
        if (!pattern) {
            return res.status(400).json({
                success: false,
                error: 'Pattern is required'
            });
        }

        console.log(`Starting serial search for pattern: ${pattern}`);

        // Check if executable exists, compile if needed
        const executablePath = path.join(SERIAL_PATH, 'serialSearch');
        if (!fs.existsSync(executablePath)) {
            console.log('Compiling serial search program...');
            await executeCommand('gcc', ['-o', 'serialSearch', 'serialSearch.c'], {
                cwd: SERIAL_PATH
            });
        }

        // Run serial search
        const result = await executeCommand('./serialSearch', [], {
            cwd: SERIAL_PATH,
            input: pattern + '\n',
            timeout: 600000 // 10 minutes
        });

        const parsedResults = parseResults(result.output);

        res.json({
            success: true,
            message: 'Serial search completed successfully',
            output: result.output,
            executionTime: parsedResults.executionTime || result.executionTime,
            totalMatches: parsedResults.totalMatches,
            matchDetails: parsedResults.details
        });

    } catch (error) {
        console.error('Serial search error:', error);
        res.status(500).json({
            success: false,
            error: 'Serial search failed',
            details: error.error || error.message,
            output: error.output || ''
        });
    }
});

// OpenMP search endpoint
app.post('/api/openmp', async (req, res) => {
    try {
        const { pattern, threads } = req.body;
        
        if (!pattern) {
            return res.status(400).json({
                success: false,
                error: 'Pattern is required'
            });
        }

        if (!threads || threads < 1) {
            return res.status(400).json({
                success: false,
                error: 'Valid thread count is required'
            });
        }

        console.log(`Starting OpenMP search for pattern: ${pattern} with ${threads} threads`);

        // Check if executable exists, compile if needed
        const executablePath = path.join(OPENMP_PATH, 'openMPsearch');
        if (!fs.existsSync(executablePath)) {
            console.log('Compiling OpenMP search program...');
            await executeCommand('gcc', ['-fopenmp', '-o', 'openMPsearch', 'openMPsearch.c'], {
                cwd: OPENMP_PATH
            });
        }

        // Run OpenMP search
        const result = await executeCommand('./openMPsearch', [], {
            cwd: OPENMP_PATH,
            input: `${pattern}\n${threads}\n`,
            timeout: 600000, // 10 minutes
            env: { ...process.env, OMP_NUM_THREADS: threads.toString() }
        });

        const parsedResults = parseResults(result.output);

        res.json({
            success: true,
            message: `OpenMP search completed successfully with ${threads} threads`,
            output: result.output,
            executionTime: parsedResults.executionTime || result.executionTime,
            totalMatches: parsedResults.totalMatches,
            matchDetails: parsedResults.details,
            threads: threads
        });

    } catch (error) {
        console.error('OpenMP search error:', error);
        res.status(500).json({
            success: false,
            error: 'OpenMP search failed',
            details: error.error || error.message,
            output: error.output || ''
        });
    }
});

// CUDA search endpoint
app.post('/api/cuda', async (req, res) => {
    try {
        const { pattern } = req.body;
        
        if (!pattern) {
            return res.status(400).json({
                success: false,
                error: 'Pattern is required'
            });
        }

        console.log(`Starting CUDA search for pattern: ${pattern}`);

        // Check if executable exists, compile if needed
        const executablePath = path.join(CUDA_PATH, 'cudaSearch');
        if (!fs.existsSync(executablePath)) {
            console.log('Compiling CUDA search program...');
            await executeCommand('nvcc', ['-o', 'cudaSearch', 'cudaSearch.cu'], {
                cwd: CUDA_PATH
            });
        }

        // Run CUDA search
        const result = await executeCommand('./cudaSearch', [], {
            cwd: CUDA_PATH,
            input: pattern + '\n',
            timeout: 600000 // 10 minutes
        });

        const parsedResults = parseResults(result.output);

        res.json({
            success: true,
            message: 'CUDA search completed successfully',
            output: result.output,
            executionTime: parsedResults.executionTime || result.executionTime,
            totalMatches: parsedResults.totalMatches,
            matchDetails: parsedResults.details
        });

    } catch (error) {
        console.error('CUDA search error:', error);
        res.status(500).json({
            success: false,
            error: 'CUDA search failed',
            details: error.error || error.message,
            output: error.output || ''
        });
    }
});

// Hybrid search endpoint
app.post('/api/hybrid', async (req, res) => {
    try {
        const { pattern, threads } = req.body;
        
        if (!pattern) {
            return res.status(400).json({
                success: false,
                error: 'Pattern is required'
            });
        }

        if (!threads || threads < 1) {
            return res.status(400).json({
                success: false,
                error: 'Valid thread count is required'
            });
        }

        console.log(`Starting hybrid search for pattern: ${pattern} with ${threads} threads`);

        // Check if executable exists
        const executablePath = path.join(HYBRID_PATH, 'hybridSearch');
        if (!fs.existsSync(executablePath)) {
            console.log('Hybrid search executable not found, compiling...');
            try {
                await executeCommand('nvcc', ['-o', 'hybridSearch', 'hybridSearch.cu', '-Xcompiler', '-fopenmp', '-Wno-deprecated-gpu-targets'], {
                    cwd: HYBRID_PATH
                });
            } catch (compileError) {
                return res.status(500).json({
                    success: false,
                    error: 'Failed to compile hybrid search program',
                    details: compileError.error || compileError.message
                });
            }
        }

        // Run hybrid search
        const result = await executeCommand('./hybridSearch', [], {
            cwd: HYBRID_PATH,
            input: `${pattern}\n${threads}\n`,
            timeout: 600000, // 10 minutes
            env: { ...process.env, OMP_NUM_THREADS: threads.toString() }
        });

        const parsedResults = parseResults(result.output);

        res.json({
            success: true,
            message: `Hybrid search completed successfully with ${threads} threads`,
            output: result.output,
            executionTime: parsedResults.executionTime || result.executionTime, // Program time (more accurate)
            serverExecutionTime: result.executionTime, // Total server time (includes overhead)
            totalMatches: parsedResults.totalMatches,
            matchDetails: parsedResults.details,
            threads: threads,
            processingRate: parsedResults.processingRate,
            chunksProcessed: parsedResults.chunksProcessed
        });

    } catch (error) {
        console.error('Hybrid search error:', error);
        res.status(500).json({
            success: false,
            error: 'Hybrid search failed',
            details: error.error || error.message,
            output: error.output || ''
        });
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        success: true,
        message: 'HPC Genome Search API is running',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
});

// System information endpoint
app.get('/api/system', async (req, res) => {
    try {
        const systemInfo = {
            nodeVersion: process.version,
            platform: process.platform,
            arch: process.arch,
            uptime: process.uptime(),
            memory: process.memoryUsage()
        };

        // Check for CUDA availability
        try {
            const cudaResult = await executeCommand('nvcc', ['--version'], { timeout: 5000 });
            systemInfo.cuda = {
                available: true,
                version: cudaResult.output
            };
        } catch (error) {
            systemInfo.cuda = {
                available: false,
                error: 'CUDA not available'
            };
        }

        // Check for OpenMP support
        try {
            const gccResult = await executeCommand('gcc', ['--version'], { timeout: 5000 });
            systemInfo.openmp = {
                available: gccResult.output.includes('gcc'),
                version: gccResult.output
            };
        } catch (error) {
            systemInfo.openmp = {
                available: false,
                error: 'GCC not available'
            };
        }

        res.json({
            success: true,
            systemInfo
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: 'Failed to get system information',
            details: error.message
        });
    }
});

// Serve the frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    res.status(500).json({
        success: false,
        error: 'Internal server error',
        details: error.message
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 HPC Genome Search Frontend Server running on http://localhost:${PORT}`);
    console.log(`📁 Project root: ${PROJECT_ROOT}`);
    console.log(`🧬 Ready to serve genome search requests!`);
});

module.exports = app;
