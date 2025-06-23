import requests
from pydantic import BaseModel

class UserInput(BaseModel):
    content: str
    
def llama_ai(user_input: UserInput) -> str:
    API_KEY = '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'  
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost",       
        "X-Title": "Terminal Llama CLI"
    }

    payload = {
        "model": "meta-llama/llama-4-maverick:free",
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content']
            deepseektext = UserInput(content=result)
            return deepseektext.content
        else:
            return f"[Error {response.status_code}]: {response.text}"
    except Exception as e:
        return f"[Exception]: {str(e)}"
    
if __name__ == "__main__":
    user_input = input("Hi: ")
    response = llama_ai(user_input)
    print(response)
    print(type(response))