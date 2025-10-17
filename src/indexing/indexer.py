import os
import json
from typing import List, Dict, Any
from tqdm import tqdm

from ..chunking.code_chunker import PythonCodeChunker
from ..chunking.doc_chunker import MarkdownChunker
from ..retrieval.bm25 import BM25Retriever


class RepositoryIndexer:
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size
        self.code_chunker = PythonCodeChunker(max_chunk_size)
        self.doc_chunker = MarkdownChunker(max_chunk_size)
        self.retriever = BM25Retriever()
        self.chunks = []

    def index_repository(self, repo_path: str,
                        output_dir: str = "data/indexes"):
        """Index entire repository"""
        print("Starting repository indexing...")

        # Find all relevant files
        files_to_index = self._find_files(repo_path)

        # Process files with progress bar
        all_chunks = []
        for file_path in tqdm(files_to_index, desc="Processing files"):
            chunks = self._process_file(file_path)
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks from {len(files_to_index)} files")

        # Index with BM25
        print("Building BM25 index...")
        self.retriever.index_documents(all_chunks)
        self.chunks = all_chunks

        # Save index to disk
        self._save_index(output_dir)
        print(f"Index saved to {output_dir}")

    def _find_files(self, repo_path: str) -> List[str]:
        """Find files to index"""
        extensions = {'.py', '.md', '.rst', '.txt', '.yaml', '.yml', '.json'}
        files = []

        for root, _, filenames in os.walk(repo_path):
            # Skip common non-source directories
            if any(skip in root for skip in
                ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
                continue

            for filename in filenames:
                if any(filename.endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, filename))

        return files

    def _process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process single file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        # Choose appropriate chunker
        if file_path.endswith('.py'):
            return self.code_chunker.chunk_content(content, file_path)
        elif file_path.endswith(('.md', '.rst')):
            return self.doc_chunker.chunk_content(content, file_path)
        else:
            # Generic text chunking
            return self.code_chunker._simple_split(content, file_path)

    def _save_index(self, output_dir: str):
        """Save index components to disk"""
        os.makedirs(output_dir, exist_ok=True)

        # Save chunks
        with open(f"{output_dir}/chunks.json", 'w') as f:
            json.dump(self.chunks, f, indent=2)

        # Save BM25 parameters
        # (simplified - you might want to pickle the full object)
        index_data = {
            'doc_freqs': dict(self.retriever.doc_freqs),
            'idf': self.retriever.idf,
            'doc_len': self.retriever.doc_len,
            'avgdl': self.retriever.avgdl,
            'k1': self.retriever.k1,
            'b': self.retriever.b
        }

        with open(f"{output_dir}/bm25_index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
