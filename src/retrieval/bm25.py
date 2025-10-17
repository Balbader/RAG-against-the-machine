import math
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
import re


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization - can be enhanced
        return re.findall(r'\b\w+\b', text.lower())

    def index_documents(self, chunks: List[Dict[str, Any]]):
        self.documents = chunks

        # Tokenize all documents
        tokenized_docs = []
        for chunk in chunks:
            tokens = self._tokenize(chunk['content'])
            tokenized_docs.append(tokens)
            self.doc_len.append(len(tokens))

        self.avgdl = sum(self.doc_len) / len(self.doc_len)

        # Calculate document frequencies
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        # Calculate IDF values
        num_docs = len(tokenized_docs)
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((num_docs - freq + 0.5) / (freq + 0.5))

        self.tokenized_docs = tokenized_docs

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        query_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self.tokenized_docs):
            score = 0
            doc_len = self.doc_len[i]

            # Count term frequencies in document
            doc_tf = Counter(doc_tokens)

            for token in query_tokens:
                if token in doc_tf:
                    tf = doc_tf[token]
                    idf = self.idf.get(token, 0)

                    # BM25 formula
                    score += idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 *
                        (1 - self.b + self.b * doc_len / self.avgdl)
                    )

            scores.append((i, score))

        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
