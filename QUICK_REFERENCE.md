# RAG System - Quick Reference Card

## 🚀 Fastest Way to Test Everything

```bash
python test_system.py
```

**Expected:** 7/7 tests passed ✅

---

## 📋 All CLI Commands

```bash
# 1. Index repository
python -m src index .

# 2. Search query
python -m src search "your question" --k 10

# 3. Generate answer (requires Ollama)
python -m src answer "your question" --k 10

# 4. Process dataset
python -m src search_dataset data/datasets/sample_questions.json

# 5. Generate dataset answers
python -m src answer_dataset data/datasets/sample_questions.json

# 6. Evaluate recall
python -m src measure_recall_at_k_on_dataset results.json truth.json
```

---

## 📊 What Each Test Verifies

| # | Capability | What It Tests |
|---|------------|---------------|
| 1 | **Indexing** | Files discovered, chunked, and indexed |
| 2 | **Chunking** | AST-based Python, header-based Markdown |
| 3 | **Retrieval** | BM25 ranking, top-k selection |
| 4 | **LLM** | Context management, Ollama integration |
| 5 | **JSON** | Pydantic validation, schema compliance |
| 6 | **CLI** | All commands work, proper I/O |
| 7 | **Metrics** | Overlap & recall@k calculation |

---

## 📈 Performance Benchmarks

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Indexing | 70 sec | < 5 min | ✅ |
| Retrieval | 1.4 sec | < 1 min | ✅ |
| Chunks | 233,454 | - | ✅ |
| Vocabulary | 570,546 | - | ✅ |

---

## 📁 Key Files

### Input
- `data/datasets/sample_questions.json` - Test questions

### Output
- `data/indexes/` - BM25 index files
- `data/output/search_results/` - Search results
- `data/output/answers/` - Generated answers
- `data/output/test_report.json` - Test results

### Documentation
- `README_TESTING.md` - Testing quick start
- `TESTING_GUIDE.md` - Detailed procedures
- `TEST_RESULTS_SUMMARY.md` - Results report
- `IMPLEMENTATION_STATUS.md` - Implementation details

---

## 🔧 Setup

```bash
# Install dependencies
pip install pydantic fire tqdm ollama

# Start Ollama (for answer generation)
ollama serve
ollama pull qwen3:0.6b

# Run tests
python test_system.py
```

---

## ✅ Success Checklist

After running tests:
- [ ] 7/7 tests passed
- [ ] Index files in `data/indexes/`
- [ ] Test report in `data/output/test_report.json`
- [ ] All CLI commands work
- [ ] Performance targets met

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Ollama error | Run `ollama serve` |
| Import error | `pip install pydantic fire tqdm ollama` |
| No index | Run `python -m src index .` first |
| Slow indexing | Normal for large repos (11k+ files) |

---

## 📞 Get Help

```bash
# CLI help
python -m src --help

# View test results
cat data/output/test_report.json

# Read full guide
cat TESTING_GUIDE.md
```

---

**🎯 One Command to Rule Them All:**

```bash
python test_system.py && echo "✅ ALL SYSTEMS GO!"
```
