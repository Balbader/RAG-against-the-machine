from ollama import chat
from typing import List, Dict, Any


class OllamaClient:
    def __init__(self, model: str = "qwen3:0.6b"):
        self.model = model

    def generate_answer(self, question: str, context_chunks:
                        List[Dict[str, Any]]) -> str:
        """Generate answer using retrieved context"""

        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks[:5]):  # Limit context size
            context_parts.append(f"Source {i+1} ({chunk['file_path']}):\n{chunk['content']}\n")

        context = "\n---\n".join(context_parts)

        # Create prompt
        prompt = f"""Based on the following context from a code repository, answer the question below.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based only on the information in the context above. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""

        try:
            response = chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating answer: {str(e)}"
