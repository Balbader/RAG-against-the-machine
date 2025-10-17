import json
import fire
from pathlib import Path
from tqdm import tqdm

from .indexing.indexer import RepositoryIndexer
from .retrieval.bm25 import BM25Retriever
from .generation.llm_client import OllamaClient
from .models.data_models import *
from .evaluation.metrics import calculate_recall_at_k, evaluate_dataset_recall


class RAGSystem:
    def __init__(self):
        self.indexer = RepositoryIndexer()
        self.retriever = BM25Retriever()
        self.llm_client = OllamaClient()
        self.chunks = []

    def index(self, repo_path: str = "."):
        """Index the repository"""
        print(f"Indexing repository at: {repo_path}")
        self.indexer.index_repository(repo_path)
        print("Indexing complete!")

    def search(self, query: str, k: int = 10):
        """Search the indexed repository"""
        self._load_index()

        # Perform search
        results = self.retriever.search(query, k)

        # Convert to required format
        retrieved_sources = []
        for doc_idx, score in results:
            chunk = self.chunks[doc_idx]
            retrieved_sources.append(MinimalSource(
                file_path=chunk['file_path'],
                first_character_index=chunk['start_char'],
                last_character_index=chunk['end_char']
            ))

        search_result = StudentSearchResults(
            search_results=[MinimalSearchResults(
                question_id="single_query",
                retrieved_sources=retrieved_sources
            )],
            k=k
        )

        # Save and print results
        output_file = "data/output/search_results/single_query.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(search_result.dict(), f, indent=2)

        print(f"Search results saved to: {output_file}")
        return search_result

    def answer(self, question: str, k: int = 10):
        """Answer single query with context"""
        self._load_index()

        # Search for relevant chunks
        results = self.retriever.search(question, k)
        context_chunks = [self.chunks[doc_idx] for doc_idx, _ in results]

        # Generate answer
        answer = self.llm_client.generate_answer(question, context_chunks)

        # Format results
        retrieved_sources = []
        for doc_idx, score in results:
            chunk = self.chunks[doc_idx]
            retrieved_sources.append(MinimalSource(
                file_path=chunk['file_path'],
                first_character_index=chunk['start_char'],
                last_character_index=chunk['end_char']
            ))

        result = StudentSearchResultsAndAnswer(
            search_results=[MinimalAnswer(
                question_id="single_query",
                retrieved_sources=retrieved_sources,
                answer=answer
            )],
            k=k
        )

        # Save results
        output_file = "data/output/answers/single_query.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(result.dict(), f, indent=2)

        print(f"Answer: {answer}")
        print(f"Results saved to: {output_file}")
        return result

    def _load_index(self):
        """Load saved index"""
        try:
            with open("data/indexes/chunks.json", 'r') as f:
                self.chunks = json.load(f)

            with open("data/indexes/bm25_index.json", 'r') as f:
                index_data = json.load(f)

            # Reconstruct BM25 index
            self.retriever.doc_freqs.update(index_data['doc_freqs'])
            self.retriever.idf = index_data['idf']
            self.retriever.doc_len = index_data['doc_len']
            self.retriever.avgdl = index_data['avgdl']
            self.retriever.k1 = index_data['k1']
            self.retriever.b = index_data['b']

            # Retokenize documents (in production, you'd save this too)
            self.retriever.documents = self.chunks
            tokenized_docs = []
            for chunk in self.chunks:
                tokens = self.retriever._tokenize(chunk['content'])
                tokenized_docs.append(tokens)
            self.retriever.tokenized_docs = tokenized_docs

        except FileNotFoundError:
            print("No index found. Please run 'uv run python -m src index' first.")
            raise

    def search_dataset(self, dataset_file: str, output_file: str = None, k: int = 10):
        """Process dataset for search evaluation"""
        self._load_index()

        with open(dataset_file, 'r') as f:
            dataset = json.load(f)

        results = []
        for question_data in tqdm(dataset['rag_questions'], desc="Processing questions"):
            question = question_data['question']
            question_id = question_data['question_id']

            # Perform search
            search_results = self.retriever.search(question, k=k)

            retrieved_sources = []
            for doc_idx, score in search_results:
                chunk = self.chunks[doc_idx]
                retrieved_sources.append({
                    'file_path': chunk['file_path'],
                    'first_character_index': chunk['start_char'],
                    'last_character_index': chunk['end_char']
                })

            results.append({
                'question_id': question_id,
                'retrieved_sources': retrieved_sources
            })

        output = StudentSearchResults(search_results=results, k=k)

        if not output_file:
            output_file = f"data/output/search_results/{Path(dataset_file).stem}.json"

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output.dict(), f, indent=2)

        print(f"Results saved to: {output_file}")
        return output

    def answer_dataset(self, dataset_file: str, output_file: str = None, k: int = 10):
        """Process dataset for answer generation"""
        self._load_index()

        with open(dataset_file, 'r') as f:
            dataset = json.load(f)

        results = []
        for question_data in tqdm(dataset['rag_questions'], desc="Generating answers"):
            question = question_data['question']
            question_id = question_data['question_id']

            # Search for relevant chunks
            search_results = self.retriever.search(question, k=k)
            context_chunks = [self.chunks[doc_idx] for doc_idx, _ in search_results]

            # Generate answer
            answer = self.llm_client.generate_answer(question, context_chunks)

            # Format results
            retrieved_sources = []
            for doc_idx, score in search_results:
                chunk = self.chunks[doc_idx]
                retrieved_sources.append({
                    'file_path': chunk['file_path'],
                    'first_character_index': chunk['start_char'],
                    'last_character_index': chunk['end_char']
                })

            results.append({
                'question_id': question_id,
                'retrieved_sources': retrieved_sources,
                'answer': answer
            })

        output = StudentSearchResultsAndAnswer(search_results=results, k=k)

        if not output_file:
            output_file = f"data/output/answers/{Path(dataset_file).stem}.json"

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output.dict(), f, indent=2)

        print(f"Results saved to: {output_file}")
        return output

    def measure_recall_at_k_on_dataset(self, search_results_file: str, ground_truth_file: str):
        """Evaluate recall@k on entire dataset"""
        recall = evaluate_dataset_recall(search_results_file, ground_truth_file)
        print(f"Recall@k: {recall:.4f} ({recall*100:.2f}%)")
        return recall


def main():
    fire.Fire(RAGSystem)


if __name__ == "__main__":
    main()
