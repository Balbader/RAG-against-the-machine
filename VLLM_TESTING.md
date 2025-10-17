# Testing RAG System on VLLM Repository

This guide explains how to test the RAG system specifically on the VLLM 0.10.1 codebase.

## Quick Start

### Single Command Test

```bash
python test_vllm.py
```

This will:

1. ✅ Index the entire VLLM 0.10.1 repository
2. ✅ Run sample searches on VLLM-specific topics
3. ✅ Generate answers about VLLM features (requires Ollama)
4. ✅ Create comprehensive test report

---

## Manual Testing Steps

### Step 1: Index VLLM Repository

```bash
python -m src index "VLLM 0.10.1/vllm-0.10.1"
```

**What this does:**

- Discovers all Python files, documentation, and configs
- Creates intelligent chunks (AST for code, headers for docs)
- Builds BM25 search index
- Saves to `data/indexes/`

**Expected output:**

```
Starting repository indexing...
Processing files: 100%|██████████| XXXX/XXXX
Created XXXXX chunks from XXXX files
Building BM25 index...
Index saved to data/indexes
```

---

### Step 2: Search VLLM Code

```bash
# Search for distributed inference info
python -m src search "How does vLLM handle distributed inference?" --k 10

# Search for PagedAttention
python -m src search "What is PagedAttention in vLLM?" --k 10

# Search for OpenAI server config
python -m src search "How to configure OpenAI compatible server?" --k 10

# Search for quantization
python -m src search "Explain vLLM's quantization support" --k 10

# Search for batching
python -m src search "How does continuous batching work?" --k 10
```

**What to verify:**

- ✅ Relevant code files retrieved
- ✅ Documentation chunks included
- ✅ BM25 scores ranking correctly
- ✅ Results include file paths and positions

---

### Step 3: Generate Answers about VLLM

**Prerequisites:**

```bash
# Ensure Ollama is running
ollama serve

# Verify model is available
ollama list | grep qwen3
```

**Generate answers:**

```bash
# Ask about vLLM features
python -m src answer "What is vLLM and what are its main features?" --k 10

# Ask about performance
python -m src answer "How does vLLM achieve high throughput?" --k 10

# Ask about architecture
python -m src answer "Explain vLLM's architecture and design" --k 10

# Ask about usage
python -m src answer "How do I deploy vLLM with OpenAI API?" --k 10
```

**What to verify:**

- ✅ Context retrieved from VLLM docs/code
- ✅ Answers reference specific VLLM features
- ✅ Citations include source files
- ✅ Technical accuracy

---

### Step 4: Process VLLM Question Dataset

Create a dataset of VLLM-specific questions:

```json
{
  "rag_questions": [
    {
      "question_id": "vllm_001",
      "question": "How does vLLM implement PagedAttention?"
    },
    {
      "question_id": "vllm_002",
      "question": "What quantization methods does vLLM support?"
    },
    {
      "question_id": "vllm_003",
      "question": "How to enable continuous batching in vLLM?"
    }
  ]
}
```

Save to `data/datasets/vllm_questions.json`, then:

```bash
# Generate search results
python -m src search_dataset data/datasets/vllm_questions.json --k 10

# Generate answers
python -m src answer_dataset data/datasets/vllm_questions.json --k 10
```

---

## VLLM-Specific Test Queries

Here are curated queries that demonstrate RAG capabilities on VLLM:

### Architecture & Design

```bash
python -m src search "PagedAttention memory management" --k 5
python -m src search "KV cache implementation details" --k 5
python -m src search "Continuous batching algorithm" --k 5
```

### Distributed Inference

```bash
python -m src search "Tensor parallelism in vLLM" --k 5
python -m src search "Pipeline parallelism configuration" --k 5
python -m src search "Multi-GPU setup and communication" --k 5
```

### Model Support

```bash
python -m src search "Supported model architectures" --k 5
python -m src search "Adding new model to vLLM" --k 5
python -m src search "Model loading and initialization" --k 5
```

### Quantization

```bash
python -m src search "AWQ quantization support" --k 5
python -m src search "GPTQ quantization implementation" --k 5
python -m src search "Quantization performance impact" --k 5
```

### Deployment

```bash
python -m src search "OpenAI API compatibility" --k 5
python -m src search "Docker deployment guide" --k 5
python -m src search "Production deployment best practices" --k 5
```

---

## Expected Results

### Indexing Performance

- **Files:** ~2,000-3,000 files (Python, Markdown, YAML, etc.)
- **Chunks:** ~50,000-100,000 chunks
- **Time:** 30-120 seconds (depends on system)
- **Vocabulary:** 100,000-200,000 unique tokens

### Search Performance

- **Retrieval time:** < 2 seconds per query
- **Relevant results:** Code + docs mixed appropriately
- **Top results:** Specific to query (not generic)

### Answer Quality

- **Context:** Drawn from VLLM source code and docs
- **Accuracy:** Technical details correct
- **Citations:** Include specific file names
- **Coherence:** Well-formed explanations

---

## Output Files

After running tests, check these locations:

### If using test_vllm.py:

```
data/vllm_tests/
├── vllm_search_results.json    # Search results
├── vllm_answers.json           # Generated answers
└── vllm_test_report.json       # Performance metrics
```

### If using CLI directly:

```
data/
├── vllm_indexes/               # VLLM index files
│   ├── chunks.json
│   └── bm25_index.json
└── output/
    ├── search_results/
    └── answers/
```

---

## Sample Questions Dataset

Create `data/datasets/vllm_questions.json`:

```json
{
  "rag_questions": [
    {
      "question_id": "vllm_001",
      "question": "How does vLLM implement PagedAttention?"
    },
    {
      "question_id": "vllm_002",
      "question": "What quantization methods does vLLM support?"
    },
    {
      "question_id": "vllm_003",
      "question": "How to enable continuous batching in vLLM?"
    },
    {
      "question_id": "vllm_004",
      "question": "How does vLLM handle distributed inference?"
    },
    {
      "question_id": "vllm_005",
      "question": "What is the OpenAI compatible server in vLLM?"
    },
    {
      "question_id": "vllm_006",
      "question": "How to configure tensor parallelism?"
    },
    {
      "question_id": "vllm_007",
      "question": "What are vLLM's key performance optimizations?"
    },
    {
      "question_id": "vllm_008",
      "question": "How does vLLM manage GPU memory?"
    },
    {
      "question_id": "vllm_009",
      "question": "What models are supported by vLLM?"
    },
    {
      "question_id": "vllm_010",
      "question": "How to deploy vLLM in production?"
    }
  ]
}
```

---

## Verification Checklist

After testing on VLLM, verify:

- [ ] **Indexing Complete**
  - [ ] All Python files indexed
  - [ ] Documentation files included
  - [ ] Config files (YAML, JSON) processed
  - [ ] Index saved to disk

- [ ] **Retrieval Working**
  - [ ] Queries return relevant results
  - [ ] Mix of code and documentation
  - [ ] Scores ranking appropriately
  - [ ] File paths correct

- [ ] **Chunking Intelligent**
  - [ ] Python functions/classes as chunks
  - [ ] Markdown sections preserved
  - [ ] Chunk sizes reasonable (<2000 chars)
  - [ ] Metadata includes positions

- [ ] **LLM Answers Generated**
  - [ ] Context retrieved correctly
  - [ ] Answers reference VLLM specifics
  - [ ] Technical accuracy maintained
  - [ ] Sources cited properly

---

## Troubleshooting

### Issue: "VLLM repository not found"

**Solution:**

```bash
# Check the folder exists
ls -la "VLLM 0.10.1/vllm-0.10.1/"

# If path is different, update test_vllm.py:
# VLLMTester(vllm_path="your/actual/path")
```

### Issue: Too many files, slow indexing

**Solution:**

```bash
# Exclude directories you don't need
# Edit src/indexing/indexer.py, add to skip list:
# 'tests', 'benchmarks', '.buildkite', etc.
```

### Issue: Ollama timeout during answer generation

**Solution:**

```bash
# Use a faster model
# Edit src/generation/llm_client.py:
# self.model = "llama3:8b"  # or another fast model

# Or increase timeout in Ollama
```

---

## Performance Expectations

### For VLLM 0.10.1 Repository

| Metric            | Expected Range   |
| ----------------- | ---------------- |
| Files to index    | 2,000-3,000      |
| Total chunks      | 50,000-100,000   |
| Indexing time     | 30-120 seconds   |
| Vocabulary size   | 100K-200K tokens |
| Search time/query | 0.5-2 seconds    |
| Answer generation | 5-30 seconds     |

---

## Advanced: Custom VLLM Evaluation

Create ground truth for VLLM-specific questions:

```json
{
  "rag_questions": [
    {
      "question_id": "vllm_001",
      "question": "How does vLLM implement PagedAttention?",
      "sources": [
        {
          "file_path": "VLLM 0.10.1/vllm-0.10.1/vllm/attention/backends/paged_attention.py",
          "first_character_index": 0,
          "last_character_index": 500
        }
      ],
      "answer": "PagedAttention is implemented..."
    }
  ]
}
```

Then measure recall:

```bash
python -m src measure_recall_at_k_on_dataset \
  data/output/search_results/vllm_questions.json \
  data/datasets/vllm_ground_truth.json
```

---

## CLI Commands Summary

```bash
# 1. Automated test (recommended)
python test_vllm.py

# 2. Manual indexing
python -m src index "VLLM 0.10.1/vllm-0.10.1"

# 3. Single search
python -m src search "your VLLM question" --k 10

# 4. Single answer
python -m src answer "your VLLM question" --k 10

# 5. Batch processing
python -m src search_dataset data/datasets/vllm_questions.json

# 6. Batch answers
python -m src answer_dataset data/datasets/vllm_questions.json

# 7. Evaluate
python -m src measure_recall_at_k_on_dataset results.json truth.json
```

---

## Success Criteria

Your RAG system on VLLM is working correctly if:

✅ Indexes the entire VLLM repository successfully
✅ Retrieves relevant code and documentation for queries
✅ Chunks Python code by functions/classes intelligently
✅ Chunks Markdown docs by headers appropriately
✅ Generates technically accurate answers about VLLM
✅ Cites specific source files in context
✅ Meets performance targets (indexing < 5 min, search < 1 min)

---

## Example Session

```bash
# Terminal session
$ python test_vllm.py

╔══════════════════════════════════════════════════════════════════════════════╗
║               RAG SYSTEM TEST ON VLLM 0.10.1 REPOSITORY                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
  TEST 1: Indexing VLLM Repository
================================================================================

Target: VLLM 0.10.1/vllm-0.10.1
This will index all Python code, documentation, and config files...
Starting repository indexing...
Processing files: 100%|██████████| 2847/2847 [00:45<00:00]
Created 87,423 chunks from 2,847 files
Building BM25 index...
Index saved to data/vllm_indexes

✓ VLLM indexing completed!
  Time: 48.23 seconds
  Chunks: 87,423
  Vocabulary: 156,789 unique tokens

================================================================================
  TEST 2: Searching VLLM Codebase
================================================================================

📝 Query: How does vLLM handle distributed inference?
   Retrieved in 843.2ms
   Top 3 results:
     1. [distributed.py] Score: 28.45
        Preview: class DistributedInference:
    """Handles distributed model inference...
     2. [parallelism.md] Score: 24.12
        Preview: # Distributed Inference in vLLM

vLLM supports tensor and pipeline...
     3. [tensor_parallel.py] Score: 22.87
        Preview: def init_tensor_parallel(world_size: int, rank: int):...

...

✓ All search results saved to data/vllm_tests/vllm_search_results.json

================================================================================
  VLLM TEST REPORT
================================================================================

📊 Indexing Statistics:
  Repository: VLLM 0.10.1
  Indexing Time: 48.23 seconds
  Total Chunks: 87,423
  Vocabulary Size: 156,789

🔍 Search Performance:
  Queries Tested: 5
  Average Search Time: 921.4ms

✅ Demonstrated on VLLM 0.10.1:
  1. Indexed large codebase efficiently
  2. Retrieved relevant code and documentation
  3. Generated contextual answers
  4. Handled technical queries successfully
```

---

## Documentation

For more information:

- General testing: `README_TESTING.md`
- Complete guide: `TESTING_GUIDE.md`

---

**Ready to test on VLLM?**

```bash
python test_vllm.py
```
