#!/bin/bash
# Quick Demo Script for RAG System

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                   RAG AGAINST THE MACHINE - QUICK DEMO                      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Index the repository
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Building Indexed Knowledge Base"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Indexing repository with intelligent chunking strategies..."
python -m src index .
echo ""
echo "✓ Index created successfully!"
echo ""

# Step 2: Single search query
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Retrieval and Ranking Demo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Searching: 'How does BM25 retrieval work?'"
python -m src search "How does BM25 retrieval work?" --k 5
echo ""
echo "✓ Search results saved to data/output/search_results/single_query.json"
echo ""

# Step 3: Generate answer with LLM
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: LLM Answer Generation with Context Management"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Generating answer: 'What is the purpose of this RAG system?'"
echo "(Note: Requires Ollama running with qwen3:0.6b model)"
python -m src answer "What is the purpose of this RAG system?" --k 5
echo ""
echo "✓ Answer saved to data/output/answers/single_query.json"
echo ""

# Step 4: Process dataset
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Batch Processing Dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Processing sample_questions.json dataset..."
python -m src search_dataset data/datasets/sample_questions.json --k 10
echo ""
echo "✓ Dataset results saved to data/output/search_results/"
echo ""

# Step 5: Show output structure
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Structured JSON Output Example"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Sample output from data/output/search_results/single_query.json:"
echo ""
head -n 20 data/output/search_results/single_query.json
echo "..."
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DEMO COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✓ Demonstrated Capabilities:"
echo "  1. ✓ Building indexed knowledge base from repository files"
echo "  2. ✓ Intelligent chunking for code and documentation"
echo "  3. ✓ Retrieval and ranking with BM25"
echo "  4. ✓ Context management for LLM"
echo "  5. ✓ Structured JSON output generation"
echo "  6. ✓ Comprehensive CLI interface"
echo "  7. ✓ Batch dataset processing"
echo ""
echo "Generated files:"
echo "  - data/indexes/           (indexed knowledge base)"
echo "  - data/output/search_results/"
echo "  - data/output/answers/"
echo ""
echo "To run comprehensive tests, execute:"
echo "  python test_system.py"
echo ""
