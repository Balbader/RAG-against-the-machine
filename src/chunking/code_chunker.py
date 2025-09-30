"""
Python Code Chunker Module

This module provides specialized chunking functionality
for Python source code files.
It uses Abstract Syntax Tree (AST) parsing to intelligently
split Python code into
semantic chunks based on functions, classes, and async functions.
When AST parsing fails due to syntax errors, it falls back to simple text-based
splitting to ensure robust handling of malformed code.
"""

import ast
from typing import List, Dict, Any
from .base import BaseChunker


class PythonCodeChunker(BaseChunker):
    def chunk_content(
        self, content: str, file_path: str
    ) -> List[Dict[str, Any]]:  # type: ignore
        chunks = []
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
                ):
                    start_line = node.lineno - 1
                    end_line = (
                        node.end_lineno if node.end_lineno else start_line + 1
                    )  # type: ignore

                    lines = content.split("\n")
                    chunk_content = "\n".join(lines[start_line:end_line])

                    if len(chunk_content) <= self.max_chunk_size:
                        chunks.append(
                            {
                                "content": chunk_content,
                                "file_path": file_path,
                                "start_char": len(
                                    "\n".join(lines[:start_line])
                                ),
                                "end_char": len("\n".join(lines[:end_line])),
                                "chunk_type": "code_block",
                            }
                        )
        except SyntaxError:
            # Fallback to simple splitting
            return self._simple_split(content, file_path)

        return chunks or self._simple_split(content, file_path)

    def _simple_split(
        self, content: str, file_path: str
    ) -> List[Dict[str, Any]]:  # type: ignore
        chunks = []
        start = 0
        while start < len(content):
            end = min(start + self.max_chunk_size, len(content))
            chunks.append(
                {
                    "content": content[start:end],
                    "file_path": file_path,
                    "start_char": start,
                    "end_char": end,
                    "chunk_type": "text",
                }
            )
            start = end
        return chunks
