import os
import requests


def ask_llm(messages):

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        return "Groq API Key not found. Check .env file."

    # 🔥 AUTO-CONVERT STRING TO MESSAGE FORMAT
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are Aero AI Assistant for drone AI system management. Be professional, concise, and helpful."
            }
        ] + messages
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    else:
        return f"Groq API Error: {data}"