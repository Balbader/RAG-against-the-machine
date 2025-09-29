from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel


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
