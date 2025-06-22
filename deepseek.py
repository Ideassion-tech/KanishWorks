import requests
from pydantic import BaseModel


class UserInput(BaseModel):
    content: str
    
def deepseek_ai(user_input: UserInput) -> str:
    API_KEY = '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'  
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost",       
        "X-Title": "Terminal DeepSeek CLI"
    }

    payload = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
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
    response = deepseek_ai(user_input)
    print(response)  # Print the response from DeepSeek AI


    
