from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3', messages=[
    {
        'role': 'user',
        'content': 'Hello, how are you?'
    }
])

print(response)
print(response['message']['content'])
