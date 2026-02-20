import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def ask_llm(messages):
    
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        else:
            return f"Groq API Error: {data}"

    except Exception as e:
        return f"LLM Error: {str(e)}"