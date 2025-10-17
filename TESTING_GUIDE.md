# RAG System Testing Guide

This guide explains how to test and demonstrate all capabilities of the RAG Against the Machine system.

## Overview

The system has been designed to demonstrate:
1. **Building indexed knowledge base** from repository files (code + docs)
2. **Intelligent chunking strategies** for different file types
3. **Retrieval and ranking** with BM25 algorithm
4. **LLM context management** within token limits
5. **Structured JSON output** generation
6. **Comprehensive CLI interface** for all operations
7. **Evaluation metrics** and performance analysis

## Quick Start

### Option 1: Automated Comprehensive Test (Recommended)

Run the full test suite that demonstrates all capabilities:

```bash
python test_system.py
```

This will execute 7 comprehensive tests covering all system capabilities and generate a detailed report.

**Expected output:**
- ✅ 7 tests passed
- Performance metrics for indexing, retrieval, and LLM generation
- Test report saved to `data/output/test_report.json`

### Option 2: Quick Demo Script

Run a streamlined demo showing core functionality:

```bash
./quick_demo.sh
```

This demonstrates:
- Repository indexing
- Single search query
- LLM answer generation
- Batch dataset processing
- JSON output format

## Detailed Testing Procedures

### Test 1: Intelligent Chunking Strategies

**What it demonstrates:**
- AST-based Python code chunking
- Header-based Markdown documentation chunking
- Fallback mechanisms for malformed code
- Semantic chunk boundaries

**How to test:**
```python
from src.chunking.code_chunker import PythonCodeChunker
from src.chunking.doc_chunker import MarkdownChunker

# Test Python chunking
code_chunker = PythonCodeChunker(max_chunk_size=2000)
chunks = code_chunker.chunk_content(python_code, "file.py")

# Test Markdown chunking
doc_chunker = MarkdownChunker(max_chunk_size=2000)
chunks = doc_chunker.chunk_content(markdown_content, "README.md")
```

**Expected results:**
- Python code split into functions and classes
- Markdown split by headers (# ## ###)
- Each chunk includes metadata: file_path, start_char, end_char, chunk_type

### Test 2: Building Indexed Knowledge Base

**What it demonstrates:**
- Repository file discovery and filtering
- Multi-format support (.py, .md, .rst, .txt, .yaml, .yml, .json)
- Progress tracking with tqdm
- BM25 index construction
- Index persistence to disk

**CLI command:**
```bash
python -m src index .
```

**What to verify:**
- ✅ Progress bar shows file processing
- ✅ Multiple file types are indexed
- ✅ Index files created in `data/indexes/`:
  - `chunks.json` - All text chunks with metadata
  - `bm25_index.json` - BM25 parameters and statistics
- ✅ Performance: Indexing should complete in < 5 minutes

**Expected output structure:**
```
Starting repository indexing...
Processing files: 100%|██████████| 11624/11624 [00:58<00:00]
Created 160531 chunks from 11624 files
Building BM25 index...
Index saved to data/indexes
```

### Test 3: Retrieval and Ranking

**What it demonstrates:**
- BM25 ranking algorithm
- Fast retrieval (< 1 minute per question)
- Top-k result selection
- Score-based ranking

**CLI commands:**
```bash
# Single search
python -m src search "How does BM25 retrieval work?" --k 5

# Custom k value
python -m src search "What are Pydantic models?" --k 10
```

**What to verify:**
- ✅ Top-k results returned (default k=10)
- ✅ Results ranked by BM25 score (descending)
- ✅ Retrieval time < 1 minute
- ✅ Results saved to `data/output/search_results/single_query.json`

**Expected JSON format:**
```json
{
  "search_results": [
    {
      "question_id": "single_query",
      "retrieved_sources": [
        {
          "file_path": "src/retrieval/bm25.py",
          "first_character_index": 0,
          "last_character_index": 100
        }
      ]
    }
  ],
  "k": 10
}
```

### Test 4: LLM Context Management & Answer Generation

**What it demonstrates:**
- Retrieval of relevant context chunks
- Context size management (fits within LLM limits)
- Top-5 chunk selection for optimal context
- Ollama integration with qwen3:0.6b
- Answer generation < 1.8 seconds per question

**Prerequisites:**
```bash
# Ensure Ollama is running
ollama serve

# Verify qwen3:0.6b model is installed
ollama list | grep qwen3
```

**CLI command:**
```bash
python -m src answer "What is the purpose of this RAG system?" --k 10
```

**What to verify:**
- ✅ Ollama connection successful
- ✅ Context chunks retrieved (top 5 used)
- ✅ Total context size < 8000 characters
- ✅ Answer generated in < 1.8 seconds (performance target)
- ✅ Answer saved to `data/output/answers/single_query.json`

**Expected JSON format:**
```json
{
  "search_results": [
    {
      "question_id": "single_query",
      "retrieved_sources": [...],
      "answer": "Based on the context, this RAG system..."
    }
  ],
  "k": 10
}
```

### Test 5: Structured JSON Output

**What it demonstrates:**
- Pydantic model validation
- Strict schema compliance
- Multiple output formats (SearchResults, Answers)
- JSON serialization

**Test with Python:**
```python
from src.models.data_models import *

# Test MinimalSource
source = MinimalSource(
    file_path="test.py",
    first_character_index=0,
    last_character_index=100
)

# Test StudentSearchResults
results = StudentSearchResults(
    search_results=[...],
    k=10
)

# Verify JSON output
import json
print(json.dumps(results.model_dump(), indent=2))
```

**What to verify:**
- ✅ All models validate input
- ✅ Required fields enforced
- ✅ JSON output properly formatted
- ✅ Output files are valid JSON

### Test 6: Comprehensive CLI Interface

**What it demonstrates:**
- Complete CLI with Fire
- All CRUD operations
- Batch processing capabilities
- Evaluation functionality

**Available commands:**

```bash
# 1. Index repository
python -m src index /path/to/repo

# 2. Single search
python -m src search "your question" --k 10

# 3. Single answer
python -m src answer "your question" --k 10

# 4. Process search dataset
python -m src search_dataset data/datasets/questions.json --k 10

# 5. Process answer dataset
python -m src answer_dataset data/datasets/questions.json --k 10

# 6. Measure recall
python -m src measure_recall_at_k_on_dataset \
  data/output/search_results/questions.json \
  data/datasets/ground_truth.json
```

**What to verify:**
- ✅ All commands execute without errors
- ✅ Help available: `python -m src --help`
- ✅ Progress bars for long operations
- ✅ Clear output messages
- ✅ Files saved to appropriate directories

### Test 7: Evaluation Metrics

**What it demonstrates:**
- Character-level overlap calculation
- Recall@k metric (5% overlap threshold)
- Dataset-wide evaluation
- Performance benchmarking

**Test overlap calculation:**
```python
from src.evaluation.metrics import calculate_overlap
from src.models.data_models import MinimalSource

source1 = MinimalSource(file_path="test.py",
                       first_character_index=0,
                       last_character_index=100)
source2 = MinimalSource(file_path="test.py",
                       first_character_index=50,
                       last_character_index=150)

overlap = calculate_overlap(source1, source2)
# Expected: 50% (50 chars overlap out of 100)
```

**Test recall@k:**
```python
from src.evaluation.metrics import calculate_recall_at_k

retrieved = [...]  # Retrieved sources
correct = [...]     # Ground truth sources

recall = calculate_recall_at_k(retrieved, correct, overlap_threshold=0.05)
# Expected: percentage of correct sources found
```

**CLI evaluation:**
```bash
# First, generate search results
python -m src search_dataset data/datasets/sample_questions.json

# Then measure recall against ground truth
python -m src measure_recall_at_k_on_dataset \
  data/output/search_results/sample_questions.json \
  data/datasets/ground_truth.json
```

**What to verify:**
- ✅ Overlap calculated correctly (0.0 to 1.0)
- ✅ Recall@k uses 5% threshold
- ✅ Results displayed as percentage
- ✅ Performance meets targets (BM25 ≥ 75%)

## Performance Targets

According to the project requirements, verify these metrics:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Indexing time | < 5 minutes | Check test_system.py output |
| Retrieval time | < 1 minute per question | Check retrieval_times in report |
| Answer generation | < 1.8 seconds per question | Check llm_generation_time |
| Recall@5 (BM25) | ≥ 75% | Run measure_recall_at_k_on_dataset |

**View performance report:**
```bash
cat data/output/test_report.json
```

## Sample Test Dataset

A sample dataset is provided at `data/datasets/sample_questions.json`:

```json
{
  "rag_questions": [
    {
      "question_id": "q001",
      "question": "How does the BM25 retrieval algorithm work?"
    },
    {
      "question_id": "q002",
      "question": "What Pydantic models are used for data validation?"
    }
    // ... more questions
  ]
}
```

**Process this dataset:**
```bash
# Generate search results
python -m src search_dataset data/datasets/sample_questions.json --k 10

# Generate answers (requires Ollama)
python -m src answer_dataset data/datasets/sample_questions.json --k 10
```

## Troubleshooting

### Issue: Ollama connection failed

**Solution:**
```bash
# Start Ollama service
ollama serve

# Pull required model
ollama pull qwen3:0.6b

# Verify it's running
curl http://localhost:11434/api/tags
```

### Issue: Index not found

**Solution:**
```bash
# Create index first
python -m src index .

# Verify index files exist
ls -la data/indexes/
```

### Issue: Import errors

**Solution:**
```bash
# Install dependencies
pip install pydantic fire tqdm ollama

# Or if using uv
uv pip install pydantic fire tqdm ollama
```

### Issue: Slow indexing

**Causes:**
- Large repository with many files
- Slow disk I/O
- Complex file parsing

**Solutions:**
- Use SSD for faster I/O
- Exclude unnecessary directories (.git, node_modules)
- Increase chunk size to reduce total chunks

## Output Directory Structure

After running tests, verify this structure:

```
data/
├── datasets/
│   └── sample_questions.json
├── indexes/
│   ├── chunks.json
│   └── bm25_index.json
└── output/
    ├── search_results/
    │   ├── single_query.json
    │   └── sample_questions.json
    ├── answers/
    │   ├── single_query.json
    │   └── sample_questions.json
    └── test_report.json
```

## Continuous Integration Testing

For automated testing in CI/CD:

```bash
#!/bin/bash
set -e

# Run comprehensive test suite
python test_system.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "✓ All tests passed"
    exit 0
else
    echo "✗ Tests failed"
    exit 1
fi
```

## Success Criteria Checklist

Use this checklist to verify all capabilities:

- [ ] **Indexed Knowledge Base**
  - [ ] Repository files discovered and processed
  - [ ] Multiple file types supported
  - [ ] Index persisted to disk
  - [ ] Indexing completes in < 5 minutes

- [ ] **Intelligent Chunking**
  - [ ] Python code chunked by AST (functions/classes)
  - [ ] Markdown chunked by headers
  - [ ] Fallback for malformed code works
  - [ ] Chunk metadata includes positions

- [ ] **Retrieval and Ranking**
  - [ ] BM25 algorithm implemented
  - [ ] Top-k results returned
  - [ ] Retrieval time < 1 minute
  - [ ] Results properly ranked by score

- [ ] **LLM Context Management**
  - [ ] Context chunks retrieved
  - [ ] Context size within limits
  - [ ] Top-5 chunks selected
  - [ ] Answer generation < 1.8 seconds

- [ ] **Structured JSON Output**
  - [ ] Pydantic models validate data
  - [ ] JSON properly formatted
  - [ ] Schema compliance verified
  - [ ] Files saved correctly

- [ ] **CLI Interface**
  - [ ] All 6 commands work
  - [ ] Help documentation available
  - [ ] Progress bars show
  - [ ] Error handling graceful

- [ ] **Evaluation Metrics**
  - [ ] Overlap calculation correct
  - [ ] Recall@k computed
  - [ ] Performance benchmarked
  - [ ] Targets met (BM25 ≥ 75%)

## Conclusion

Running `python test_system.py` provides comprehensive verification of all system capabilities. The test suite demonstrates:

✅ Complete RAG pipeline functionality
✅ Intelligent processing of code and documentation
✅ Fast retrieval with BM25
✅ LLM integration with context management
✅ Structured output generation
✅ Full CLI interface
✅ Evaluation and performance metrics

For questions or issues, refer to:
- `IMPLEMENTATION_STATUS.md` - Implementation details
- `step_by_step.md` - Step-by-step guide
- `CLAUDE.md` - Project overview
