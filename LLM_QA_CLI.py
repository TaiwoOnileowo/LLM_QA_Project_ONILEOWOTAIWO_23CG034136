import string
import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables if available
load_dotenv()

def preprocess_input(text):
    """
    Applies basic preprocessing: lowercasing, punctuation removal, and tokenization.
    Returns a tuple of (processed_text_string, tokens_list).
    """
    # Lowercasing
    text_lower = text.lower()
    
    # Punctuation removal
    text_no_punct = text_lower.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization (simple whitespace split)
    tokens = text_no_punct.split()
    
    return text_no_punct, tokens

def query_llm(prompt, api_key=None, model="gpt-4o"):
    """
    Sends the prompt to GitHub Models API.
    Uses the same endpoint as your TypeScript code.
    
    Args:
        prompt: The user's question/prompt
        api_key: GitHub token (defaults to GITHUB_TOKEN env var)
        model: Model to use (default: gpt-4o, alternatives: gpt-4o-mini, etc.)
    """
    
    # GitHub Models endpoint (same as in your TS code)
    API_URL = "https://models.inference.ai.azure.com/chat/completions"
    
    if not api_key:
        # Check environment variable
        api_key = os.getenv("GITHUB_TOKEN")

    if not api_key:
        return "Error: No API Key provided. Please set GITHUB_TOKEN in .env or provide it."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides clear and accurate answers to questions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": model,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 1000
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Parse the response (GitHub Models uses OpenAI-compatible format)
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        elif "error" in data:
            return f"API Error: {data['error']}"
        else:
            return f"Unexpected response format: {data}"
            
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    except json.JSONDecodeError as e:
        return f"Failed to parse response: {e}"

def main():
    print("--- NLP Question-Answering System (CLI) ---")
    print("Using GitHub Models API (compatible with your TypeScript code)")
    print("Type 'exit' or 'quit' to stop.")
    
    # Optional: Test API connection
    api_key = os.getenv("GITHUB_TOKEN")
    if not api_key:
        print("\n⚠️  Warning: GITHUB_TOKEN not found in environment variables!")
        print("Please add it to your .env file or set it as an environment variable.")
    
    while True:
        user_input = input("\nEnter your question: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue

        # Preprocessing
        processed_text, tokens = preprocess_input(user_input)
        print(f"\n[Processed]: {processed_text}")
        print(f"[Tokens]: {tokens}")
        
        # Query LLM using GitHub Models
        print("\nQuerying GitHub Models API...")
        answer = query_llm(user_input)
        
        print(f"\n[LLM Answer]:\n{answer}")

if __name__ == "__main__":
    main()