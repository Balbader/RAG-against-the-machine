# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simple RAG (Retrieval-Augmented Generation) demonstration project that combines:
- **Ollama** for local LLM inference using the `qwen3:0.6b` model
- **BM25S** for document retrieval using BM25+ ranking
- **Pydantic** for data validation and modeling

The project demonstrates a basic RAG pipeline with document retrieval and LLM-based response generation.

## Development Environment

### Virtual Environment Setup
The project uses a Python virtual environment located at `.venv/`. To activate:
```bash
source .venv/bin/activate
```

### Required Dependencies
The main dependencies are imported in `main.py`:
- `ollama` - For LLM chat interface
- `bm25s` - For BM25 document retrieval
- `pydantic` - For data validation

Install dependencies by running the main script and checking for import errors.

### Running the Application
Execute the main application with:
```bash
python main.py
```

This will:
1. Create a simple User model with Pydantic
2. Send a chat request to the local Ollama model (`qwen3:0.6b`)
3. Demonstrate BM25 retrieval with a small corpus
4. Display results including retrieved documents and similarity scores

## Project Structure

- `main.py` - Main application entry point containing the RAG demonstration
- `.venv/` - Python virtual environment
- `VLLM 0.10.1/` - Contains VLLM source code (version 0.10.1)
- `RAG_against_the_machine.pdf` - Project documentation

## Architecture Notes

The current implementation is a minimal proof-of-concept that demonstrates:

1. **LLM Integration**: Direct integration with Ollama for local model inference
2. **Retrieval System**: Basic BM25 implementation for document similarity search
3. **Data Models**: Simple Pydantic models for structured data handling

The architecture is designed for experimentation and can be extended to include more sophisticated RAG patterns, document preprocessing, and vector-based retrieval systems.

## Development Notes

- The project currently uses a hardcoded corpus for BM25 demonstration
- Ollama model responses include thinking tokens and detailed metadata
- BM25S provides progress bars during indexing and retrieval operations