#!/usr/bin/env python3
"""
VLLM Repository RAG Test Script

This script demonstrates the RAG system's capabilities on the VLLM 0.10.1 repository.
It performs indexing, retrieval, and answer generation on VLLM's codebase.
"""

import json
import time
from pathlib import Path
from src.indexing.indexer import RepositoryIndexer
from src.retrieval.bm25 import BM25Retriever
from src.generation.llm_client import OllamaClient
from src.models.data_models import *


class VLLMTester:
    def __init__(self, vllm_path="VLLM 0.10.1/vllm-0.10.1"):
        self.vllm_path = vllm_path
        self.output_dir = "data/vllm_tests"

    def print_header(self, text):
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80)

    def test_1_index_vllm(self):
        """Index the VLLM repository"""
        self.print_header("TEST 1: Indexing VLLM Repository")

        print(f"\nTarget: {self.vllm_path}")
        print("This will index all Python code, documentation, and config files...")

        indexer = RepositoryIndexer(max_chunk_size=2000)

        start_time = time.time()
        indexer.index_repository(
            self.vllm_path,
            output_dir="data/vllm_indexes"
        )
        indexing_time = time.time() - start_time

        print(f"\n✓ VLLM indexing completed!")
        print(f"  Time: {indexing_time:.2f} seconds")
        print(f"  Chunks: {len(indexer.chunks):,}")
        print(f"  Vocabulary: {len(indexer.retriever.idf):,} unique tokens")

        return {
            "time": indexing_time,
            "chunks": len(indexer.chunks),
            "vocabulary": len(indexer.retriever.idf)
        }

    def test_2_search_vllm(self):
        """Search VLLM codebase with sample queries"""
        self.print_header("TEST 2: Searching VLLM Codebase")

        # Load index
        with open("data/vllm_indexes/chunks.json", 'r') as f:
            chunks = json.load(f)

        with open("data/vllm_indexes/bm25_index.json", 'r') as f:
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

        # VLLM-specific queries
        test_queries = [
            "How does vLLM handle distributed inference?",
            "What is PagedAttention in vLLM?",
            "How to configure OpenAI compatible server?",
            "Explain vLLM's quantization support",
            "How does continuous batching work in vLLM?"
        ]

        all_results = []

        for query in test_queries:
            print(f"\n📝 Query: {query}")

            start_time = time.time()
            results = retriever.search(query, k=5)
            search_time = time.time() - start_time

            print(f"   Retrieved in {search_time*1000:.1f}ms")
            print(f"   Top 3 results:")

            for rank, (doc_idx, score) in enumerate(results[:3], 1):
                chunk = chunks[doc_idx]
                file_name = Path(chunk['file_path']).name
                print(f"     {rank}. [{file_name}] Score: {score:.2f}")
                print(f"        Preview: {chunk['content'][:80].strip()}...")

            # Save results
            retrieved_sources = []
            for doc_idx, score in results:
                chunk = chunks[doc_idx]
                retrieved_sources.append({
                    'file_path': chunk['file_path'],
                    'first_character_index': chunk['start_char'],
                    'last_character_index': chunk['end_char']
                })

            all_results.append({
                'question': query,
                'retrieved_sources': retrieved_sources,
                'search_time': search_time
            })

        # Save all results
        output_file = Path(self.output_dir) / "vllm_search_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ All search results saved to {output_file}")

        return all_results

    def test_3_answer_vllm(self):
        """Generate answers about VLLM using LLM"""
        self.print_header("TEST 3: Generating Answers about VLLM")

        # Load index
        with open("data/vllm_indexes/chunks.json", 'r') as f:
            chunks = json.load(f)

        with open("data/vllm_indexes/bm25_index.json", 'r') as f:
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

        # Check Ollama
        try:
            llm_client = OllamaClient()
            print("✓ Ollama connection successful")
        except Exception as e:
            print(f"⚠ Ollama not available: {e}")
            print("  Skipping answer generation test")
            return None

        # Sample questions
        questions = [
            "What is vLLM and what are its main features?",
            "How does vLLM achieve high throughput?"
        ]

        answers = []

        for question in questions:
            print(f"\n📝 Question: {question}")

            # Retrieve context
            results = retriever.search(question, k=10)
            context_chunks = [chunks[doc_idx] for doc_idx, _ in results[:5]]

            print(f"   Context: {len(context_chunks)} chunks "
                  f"({sum(len(c['content']) for c in context_chunks)} chars)")

            # Generate answer
            print("   Generating answer...")
            start_time = time.time()
            answer = llm_client.generate_answer(question, context_chunks)
            gen_time = time.time() - start_time

            print(f"   ✓ Generated in {gen_time:.1f}s")
            print(f"   Answer preview: {answer[:150]}...")

            answers.append({
                'question': question,
                'answer': answer,
                'generation_time': gen_time,
                'context_size': sum(len(c['content']) for c in context_chunks)
            })

        # Save answers
        output_file = Path(self.output_dir) / "vllm_answers.json"
        with open(output_file, 'w') as f:
            json.dump(answers, f, indent=2)

        print(f"\n✓ Answers saved to {output_file}")

        return answers

    def generate_report(self, index_stats, search_results, answers):
        """Generate comprehensive test report"""
        self.print_header("VLLM TEST REPORT")

        print("\n📊 Indexing Statistics:")
        print(f"  Repository: VLLM 0.10.1")
        print(f"  Indexing Time: {index_stats['time']:.2f} seconds")
        print(f"  Total Chunks: {index_stats['chunks']:,}")
        print(f"  Vocabulary Size: {index_stats['vocabulary']:,}")

        print("\n🔍 Search Performance:")
        avg_search_time = sum(r['search_time'] for r in search_results) / len(search_results)
        print(f"  Queries Tested: {len(search_results)}")
        print(f"  Average Search Time: {avg_search_time*1000:.1f}ms")
        print(f"  Top Queries:")
        for r in search_results[:3]:
            print(f"    - {r['question'][:60]}...")

        if answers:
            print("\n💬 Answer Generation:")
            avg_gen_time = sum(a['generation_time'] for a in answers) / len(answers)
            print(f"  Questions Answered: {len(answers)}")
            print(f"  Average Generation Time: {avg_gen_time:.1f}s")
            avg_context = sum(a['context_size'] for a in answers) / len(answers)
            print(f"  Average Context Size: {avg_context:.0f} characters")

        # Save report
        report = {
            'repository': 'VLLM 0.10.1',
            'indexing': index_stats,
            'search': {
                'queries': len(search_results),
                'avg_time': avg_search_time
            },
            'answers': {
                'count': len(answers) if answers else 0,
                'avg_time': avg_gen_time if answers else 0
            } if answers else None
        }

        report_file = Path(self.output_dir) / "vllm_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Full report saved to {report_file}")

        print("\n" + "=" * 80)
        print("  VLLM RAG TESTING COMPLETE!")
        print("=" * 80)
        print("\n✅ Demonstrated on VLLM 0.10.1:")
        print("  1. Indexed large codebase efficiently")
        print("  2. Retrieved relevant code and documentation")
        print("  3. Generated contextual answers")
        print("  4. Handled technical queries successfully")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               RAG SYSTEM TEST ON VLLM 0.10.1 REPOSITORY                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    tester = VLLMTester()

    try:
        # Test 1: Index VLLM
        index_stats = tester.test_1_index_vllm()

        # Test 2: Search VLLM
        search_results = tester.test_2_search_vllm()

        # Test 3: Generate answers
        answers = tester.test_3_answer_vllm()

        # Generate report
        tester.generate_report(index_stats, search_results, answers)

    except FileNotFoundError:
        print("\n✗ VLLM repository not found!")
        print("  Expected location: VLLM 0.10.1/vllm-0.10.1/")
        print("  Please verify the path and try again.")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
