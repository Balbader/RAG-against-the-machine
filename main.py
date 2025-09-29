from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='qwen3:0.6b', messages=[
    {
        'role': 'user',
        'content': 'Hello, how are you today?'
    }
])

print(response)
print(response['message']['content'])
