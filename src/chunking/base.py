from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk_content(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Return list of chunks with metadata"""
        pass
