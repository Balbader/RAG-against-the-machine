"""
Documentation Chunker Module

This module implements chunking for documentation-like text, with a focus on
Markdown. It segments content semantically by headers and further splits large
sections by sentences to respect a maximum chunk size. Each chunk is annotated
with positional metadata for downstream retrieval and indexing.
"""

import re
from typing import List, Dict, Any
from .base import BaseChunker


class MarkdownChunker(BaseChunker):
    def chunk_content(
        self, content: str, file_path: str
    ) -> List[Dict[str, Any]]:  # type: ignore
        chunks = []

        # Split by headers
        header_pattern = r"^(#{1,6})\s+(.+)$"
        lines = content.split("\n")
        current_chunk = []
        current_start = 0

        for i, line in enumerate(lines):
            if re.match(header_pattern, line) and current_chunk:
                # Process current chunk
                chunk_content = "\n".join(current_chunk)
                if len(chunk_content.strip()) > 0:
                    chunks.extend(
                        self._process_chunk(
                            chunk_content,
                            file_path,
                            current_start,
                            len("\n".join(lines[:i])),
                        )
                    )
                current_chunk = [line]
                current_start = len("\n".join(lines[:i]))
            else:
                current_chunk.append(line)

        # Process final chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            chunks.extend(
                self._process_chunk(
                    chunk_content, file_path, current_start, len(content)
                )
            )

        return chunks

    def _process_chunk(
        self, content: str, file_path: str, start_char: int, end_char: int
    ) -> List[Dict[str, Any]]:
        if len(content) <= self.max_chunk_size:
            return [
                {
                    "content": content,
                    "file_path": file_path,
                    "start_char": start_char,
                    "end_char": end_char,
                    "chunk_type": "documentation",
                }
            ]

        # Split large sections by sentences
        sentences = re.split(r"(?<=[.!?])\s+", content)
        chunks = []
        current_chunk = ""
        current_start = start_char

        for sentence in sentences:
            if (
                len(current_chunk + sentence) > self.max_chunk_size
                and current_chunk
            ):
                chunks.append(
                    {
                        "content": current_chunk.strip(),
                        "file_path": file_path,
                        "start_char": current_start,
                        "end_char": current_start + len(current_chunk),
                        "chunk_type": "documentation",
                    }
                )
                current_start += len(current_chunk)
                current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "

        if current_chunk.strip():
            chunks.append(
                {
                    "content": current_chunk.strip(),
                    "file_path": file_path,
                    "start_char": current_start,
                    "end_char": end_char,
                    "chunk_type": "documentation",
                }
            )

        return chunks
