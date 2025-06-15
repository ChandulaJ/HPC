
# Genome Pattern Search Accelerator

This project provides a high-performance system for searching DNA patterns in large genome FASTA files. The system supports serial and parallel pattern search methods using **OpenMP**, **CUDA**, and a **hybrid model** that combines both technologies. It is optimized for efficient genomic data handling and performance comparison across different computing models.

---

## 🧬 Overview

Given a genome FASTA file (such as `Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz`), this tool allows users to:

- Provide a **FASTA file input path** (downloaded genome file).
- Provide a **DNA pattern** to search within the genome.
- Automatically **preprocess the FASTA** to identify chromosomes and line numbers for accurate indexing.
- Search the pattern using:
  - ✅ Serial algorithm
  - ✅ Parallel algorithm using **OpenMP**
  - ✅ Parallel algorithm using **CUDA**
  - ✅ **Hybrid** parallel search using both CUDA and OpenMP

---

## ⚙️ Features

- Efficient parsing of large `.fa` files
- Chromosome-wise indexing for scalable searches
- Comparative runtime measurement between different execution models
- Easy integration and input handling

---


