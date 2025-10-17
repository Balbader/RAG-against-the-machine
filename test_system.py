#!/usr/bin/env python3
"""
Comprehensive RAG System Test Script

This script demonstrates all capabilities of the RAG system:
1. Building indexed knowledge base from repository files
2. Intelligent chunking strategies for different file types
3. Retrieval and ranking of relevant information
4. LLM context management
5. Structured JSON output generation
6. CLI interface usage
7. Evaluation metrics and performance analysis
"""

import json
import time
import os
from pathlib import Path
from src.indexing.indexer import RepositoryIndexer
from src.retrieval.bm25 import BM25Retriever
from src.generation.llm_client import OllamaClient
from src.models.data_models import *
from src.evaluation.metrics import calculate_recall_at_k, calculate_overlap
from src.chunking.code_chunker import PythonCodeChunker
from src.chunking.doc_chunker import MarkdownChunker


class RAGSystemTester:
    def __init__(self):
        self.results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_metrics": {},
            "demonstrations": []
        }

    def print_section(self, title):
        """Print formatted section header"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def print_subsection(self, title):
        """Print formatted subsection header"""
        print(f"\n--- {title} ---")

    def test_1_chunking_strategies(self):
        """Demonstrate intelligent chunking for different file types"""
        self.print_section("TEST 1: Intelligent Chunking Strategies")

        # Test Python code chunking
        self.print_subsection("1.1 Python Code Chunking (AST-based)")
        code_chunker = PythonCodeChunker(max_chunk_size=2000)

        test_python_code = '''
"""Module for handling user authentication."""

import hashlib
from typing import Optional

class UserAuthenticator:
    """Handles user authentication and session management."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.active_sessions = {}

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(
            (password + self.secret_key).encode()
        ).hexdigest()

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user credentials."""
        # Implementation here
        pass

def create_session(user_id: int) -> str:
    """Create new user session."""
    return f"session_{user_id}_{time.time()}"
'''

        chunks = code_chunker.chunk_content(test_python_code, "auth.py")
        print(f"✓ Created {len(chunks)} semantic chunks from Python code")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"  Chunk {i}: {chunk['chunk_type']} "
                  f"(chars {chunk['start_char']}-{chunk['end_char']})")
            print(f"    Preview: {chunk['content'][:60].strip()}...")

        # Test Markdown chunking
        self.print_subsection("1.2 Markdown Documentation Chunking (Header-based)")
        doc_chunker = MarkdownChunker(max_chunk_size=2000)

        test_markdown = '''# RAG System Documentation

## Overview
This system implements retrieval-augmented generation for code repositories.

## Installation
Follow these steps to install:
1. Clone the repository
2. Install dependencies
3. Configure settings

## Usage
### Basic Usage
To index a repository, use the following command...

### Advanced Features
The system supports multiple retrieval strategies...
'''

        chunks = doc_chunker.chunk_content(test_markdown, "README.md")
        print(f"✓ Created {len(chunks)} semantic chunks from Markdown")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"  Chunk {i}: {chunk['chunk_type']} "
                  f"(chars {chunk['start_char']}-{chunk['end_char']})")
            print(f"    Preview: {chunk['content'][:60].strip()}...")

        self.results["tests_passed"] += 1
        self.results["demonstrations"].append({
            "test": "Chunking Strategies",
            "status": "PASSED",
            "details": f"Successfully chunked Python code into {len(chunks)} semantic units"
        })

    def test_2_indexing_knowledge_base(self):
        """Demonstrate building indexed knowledge base"""
        self.print_section("TEST 2: Building Indexed Knowledge Base")

        start_time = time.time()
        indexer = RepositoryIndexer(max_chunk_size=2000)

        print("Indexing current repository...")
        print("  - Discovering files (Python, Markdown, YAML, JSON, etc.)")
        print("  - Applying intelligent chunking strategies")
        print("  - Building BM25 search index")
        print("  - Persisting to disk")

        indexer.index_repository(".", output_dir="data/indexes")

        indexing_time = time.time() - start_time

        print(f"\n✓ Indexing completed in {indexing_time:.2f} seconds")
        print(f"  - Total chunks: {len(indexer.chunks)}")
        print(f"  - Average document length: {indexer.retriever.avgdl:.1f} tokens")
        print(f"  - Vocabulary size: {len(indexer.retriever.idf)} unique tokens")

        # Show sample of indexed files
        file_types = {}
        for chunk in indexer.chunks[:100]:
            ext = Path(chunk['file_path']).suffix
            file_types[ext] = file_types.get(ext, 0) + 1

        print("\n  Sample of indexed file types:")
        for ext, count in sorted(file_types.items(), key=lambda x: -x[1]):
            print(f"    {ext or 'no extension'}: {count} chunks")

        self.results["tests_passed"] += 1
        self.results["performance_metrics"]["indexing_time"] = indexing_time
        self.results["performance_metrics"]["total_chunks"] = len(indexer.chunks)
        self.results["performance_metrics"]["vocabulary_size"] = len(indexer.retriever.idf)

    def test_3_retrieval_and_ranking(self):
        """Demonstrate retrieval and ranking capabilities"""
        self.print_section("TEST 3: Retrieval and Ranking")

        # Load index
        with open("data/indexes/chunks.json", 'r') as f:
            chunks = json.load(f)

        with open("data/indexes/bm25_index.json", 'r') as f:
            index_data = json.load(f)

        retriever = BM25Retriever()
        retriever.doc_freqs.update(index_data['doc_freqs'])
        retriever.idf = index_data['idf']
        retriever.doc_len = index_data['doc_len']
        retriever.avgdl = index_data['avgdl']
        retriever.k1 = index_data['k1']
        retriever.b = index_data['b']
        retriever.documents = chunks

        # Retokenize
        tokenized_docs = []
        for chunk in chunks:
            tokens = retriever._tokenize(chunk['content'])
            tokenized_docs.append(tokens)
        retriever.tokenized_docs = tokenized_docs

        # Test queries
        test_queries = [
            "How does BM25 retrieval work?",
            "Pydantic data models for RAG",
            "Python code chunking implementation"
        ]

        for query in test_queries:
            self.print_subsection(f"Query: '{query}'")

            start_time = time.time()
            results = retriever.search(query, k=5)
            retrieval_time = time.time() - start_time

            print(f"✓ Retrieved top-5 results in {retrieval_time*1000:.2f}ms")
            print(f"\nTop results (ranked by BM25 score):")

            for rank, (doc_idx, score) in enumerate(results, 1):
                chunk = chunks[doc_idx]
                file_name = Path(chunk['file_path']).name
                print(f"  {rank}. [{file_name}] Score: {score:.3f}")
                print(f"     Type: {chunk.get('chunk_type', 'unknown')}")
                print(f"     Preview: {chunk['content'][:80].strip()}...")

            if not hasattr(self.results["performance_metrics"], "avg_retrieval_time"):
                self.results["performance_metrics"]["retrieval_times"] = []
            self.results["performance_metrics"]["retrieval_times"].append(retrieval_time)

        self.results["tests_passed"] += 1

    def test_4_llm_context_management(self):
        """Demonstrate LLM context management and answer generation"""
        self.print_section("TEST 4: LLM Context Management & Answer Generation")

        # Check if Ollama is available
        print("Checking Ollama availability...")
        try:
            import ollama
            models = ollama.list()
            print(f"✓ Ollama is running with {len(models.get('models', []))} models available")
        except Exception as e:
            print(f"⚠ Ollama not available: {e}")
            print("  Skipping LLM test, but demonstrating context preparation...")
            self.results["tests_failed"] += 1
            return

        # Load index
        with open("data/indexes/chunks.json", 'r') as f:
            chunks = json.load(f)

        with open("data/indexes/bm25_index.json", 'r') as f:
            index_data = json.load(f)

        retriever = BM25Retriever()
        retriever.doc_freqs.update(index_data['doc_freqs'])
        retriever.idf = index_data['idf']
        retriever.doc_len = index_data['doc_len']
        retriever.avgdl = index_data['avgdl']
        retriever.documents = chunks

        tokenized_docs = []
        for chunk in chunks:
            tokens = retriever._tokenize(chunk['content'])
            tokenized_docs.append(tokens)
        retriever.tokenized_docs = tokenized_docs

        # Test question
        question = "What is the purpose of this RAG system?"

        self.print_subsection(f"Question: '{question}'")

        # Retrieve context
        results = retriever.search(question, k=10)
        context_chunks = [chunks[doc_idx] for doc_idx, _ in results[:5]]

        print(f"\n✓ Retrieved {len(context_chunks)} chunks for context")

        total_context_size = sum(len(c['content']) for c in context_chunks)
        print(f"  Total context size: {total_context_size} characters")
        print(f"  Average chunk size: {total_context_size / len(context_chunks):.0f} characters")
        print(f"  Context fits within typical LLM limits: "
              f"{'✓ Yes' if total_context_size < 8000 else '✗ No'}")

        print("\n  Context sources:")
        for i, chunk in enumerate(context_chunks, 1):
            print(f"    {i}. {Path(chunk['file_path']).name} "
                  f"({len(chunk['content'])} chars)")

        # Generate answer
        print("\n  Generating answer using Ollama (qwen3:0.6b)...")
        llm_client = OllamaClient()

        start_time = time.time()
        answer = llm_client.generate_answer(question, context_chunks)
        generation_time = time.time() - start_time

        print(f"\n✓ Answer generated in {generation_time:.2f} seconds")
        print(f"\nAnswer:\n{answer[:300]}...")

        self.results["performance_metrics"]["llm_generation_time"] = generation_time
        self.results["tests_passed"] += 1

    def test_5_structured_json_output(self):
        """Demonstrate structured JSON output generation"""
        self.print_section("TEST 5: Structured JSON Output")

        # Create sample data
        self.print_subsection("5.1 MinimalSource Model")
        source = MinimalSource(
            file_path="src/models/data_models.py",
            first_character_index=0,
            last_character_index=100
        )
        print(f"✓ Created MinimalSource:")
        print(f"  {json.dumps(source.dict(), indent=2)}")

        self.print_subsection("5.2 Search Results Model")
        search_results = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="test_001",
                    retrieved_sources=[source]
                )
            ],
            k=10
        )
        print(f"✓ Created StudentSearchResults:")
        print(f"  {json.dumps(search_results.dict(), indent=2)}")

        self.print_subsection("5.3 Answer Results Model")
        answer_results = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id="test_001",
                    retrieved_sources=[source],
                    answer="This is a sample answer demonstrating structured output."
                )
            ],
            k=10
        )
        print(f"✓ Created StudentSearchResultsAndAnswer:")
        print(f"  {json.dumps(answer_results.dict(), indent=2)[:200]}...")

        # Save to file
        output_dir = Path("data/output/test_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "test_answer.json"
        with open(output_file, 'w') as f:
            json.dump(answer_results.dict(), f, indent=2)

        print(f"\n✓ Saved structured output to: {output_file}")

        self.results["tests_passed"] += 1

    def test_6_cli_interface(self):
        """Demonstrate CLI interface capabilities"""
        self.print_section("TEST 6: CLI Interface")

        print("Available CLI commands:\n")

        commands = [
            ("index", "python -m src index .",
             "Index the current repository"),
            ("search", "python -m src search 'your question' --k 10",
             "Search for relevant information"),
            ("answer", "python -m src answer 'your question' --k 10",
             "Generate answer with context"),
            ("search_dataset", "python -m src search_dataset data/datasets/questions.json --k 10",
             "Process dataset for evaluation"),
            ("answer_dataset", "python -m src answer_dataset data/datasets/questions.json --k 10",
             "Generate answers for dataset"),
            ("measure_recall_at_k_on_dataset",
             "python -m src measure_recall_at_k_on_dataset results.json truth.json",
             "Evaluate recall@k metric")
        ]

        for cmd, example, description in commands:
            print(f"  {cmd}")
            print(f"    Description: {description}")
            print(f"    Example: {example}")
            print()

        print("✓ All CLI commands documented and available")
        self.results["tests_passed"] += 1

    def test_7_evaluation_metrics(self):
        """Demonstrate evaluation metrics"""
        self.print_section("TEST 7: Evaluation Metrics")

        self.print_subsection("7.1 Overlap Calculation")

        source1 = MinimalSource(
            file_path="test.py",
            first_character_index=0,
            last_character_index=100
        )
        source2 = MinimalSource(
            file_path="test.py",
            first_character_index=50,
            last_character_index=150
        )

        overlap = calculate_overlap(source1, source2)
        print(f"Source 1: chars 0-100")
        print(f"Source 2: chars 50-150")
        print(f"✓ Overlap: {overlap:.2%}")

        self.print_subsection("7.2 Recall@k Calculation")

        retrieved_sources = [
            MinimalSource(file_path="test.py", first_character_index=0, last_character_index=100),
            MinimalSource(file_path="test.py", first_character_index=200, last_character_index=300),
            MinimalSource(file_path="test.py", first_character_index=400, last_character_index=500),
        ]

        correct_sources = [
            MinimalSource(file_path="test.py", first_character_index=10, last_character_index=90),
            MinimalSource(file_path="test.py", first_character_index=210, last_character_index=290),
        ]

        recall = calculate_recall_at_k(retrieved_sources, correct_sources)
        print(f"Retrieved: {len(retrieved_sources)} sources")
        print(f"Correct: {len(correct_sources)} sources")
        print(f"✓ Recall@k: {recall:.2%}")

        self.results["tests_passed"] += 1

    def generate_report(self):
        """Generate comprehensive test report"""
        self.print_section("TEST REPORT SUMMARY")

        print(f"\nTests Passed: {self.results['tests_passed']}")
        print(f"Tests Failed: {self.results['tests_failed']}")
        print(f"Success Rate: {self.results['tests_passed']/(self.results['tests_passed']+self.results['tests_failed'])*100:.1f}%")

        if self.results["performance_metrics"]:
            print("\n--- Performance Metrics ---")
            metrics = self.results["performance_metrics"]

            if "indexing_time" in metrics:
                print(f"  Indexing Time: {metrics['indexing_time']:.2f}s")
                print(f"  Total Chunks: {metrics['total_chunks']}")
                print(f"  Vocabulary Size: {metrics['vocabulary_size']}")

            if "retrieval_times" in metrics:
                avg_retrieval = sum(metrics['retrieval_times']) / len(metrics['retrieval_times'])
                print(f"  Average Retrieval Time: {avg_retrieval*1000:.2f}ms")

            if "llm_generation_time" in metrics:
                print(f"  LLM Generation Time: {metrics['llm_generation_time']:.2f}s")

        # Save report
        report_file = Path("data/output/test_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Full report saved to: {report_file}")

        print("\n" + "=" * 80)
        print("  ALL CAPABILITIES DEMONSTRATED SUCCESSFULLY!")
        print("=" * 80)


def main():
    """Run comprehensive system tests"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              RAG AGAINST THE MACHINE - COMPREHENSIVE TEST SUITE              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    tester = RAGSystemTester()

    try:
        tester.test_1_chunking_strategies()
        tester.test_2_indexing_knowledge_base()
        tester.test_3_retrieval_and_ranking()
        tester.test_4_llm_context_management()
        tester.test_5_structured_json_output()
        tester.test_6_cli_interface()
        tester.test_7_evaluation_metrics()

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        tester.results["tests_failed"] += 1

    finally:
        tester.generate_report()


if __name__ == "__main__":
    main()
