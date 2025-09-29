from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel
import bm25s


class User(BaseModel):
    id: int
    name: str


u = User(id=1, name="basil")
response: ChatResponse = chat(model='qwen3:0.6b', messages=[
    {
        'role': 'user',
        'content': 'Hello, how are you today?'
    }
])

print(u)
print(response)
print(response['message']['content'])

corpus = ["cat purrs", "dog barks", "bird sings"]
tokens = bm25s.tokenize(corpus)

retriever = bm25s.BM25(corpus=corpus, method="bm25+")
retriever.index(tokens)

docs, scores = retriever.retrieve(bm25s.tokenize("cat"), k=2)
print(docs, scores)
