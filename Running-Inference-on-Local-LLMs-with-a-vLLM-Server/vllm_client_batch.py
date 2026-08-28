#!/usr/bin/env python
# vLLM client sending an array of prompts concurrently.
#
# Usage:
# Once you have a vLLM server running, set the VLLM_HOST environment variable to the server's hostname and port.
# Then launch this script with,
# ./vllm_client_batch.py

import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

HOST = os.environ.get("VLLM_HOST", "http://127.0.0.1:8000")
API_KEY = os.environ.get("VLLM_API_KEY", "your-secret-key")

# Array of prompts
PROMPTS = [
    "Tell me about Sri Lanka",
    "Why is the sky blue?",
    "Write a Python code that calculates the Fibonacci sequence up to 15.",
]

def main() -> int:
    client = OpenAI(base_url=f"{HOST.rstrip('/')}/v1", api_key=API_KEY)

    try:
        resp = client.models.list()
        if not resp.data:
            print(f"No models served at {HOST}.")
            return 1
    except Exception as e:
        print(f"Could not list models from {HOST}: {e}")
        return 1

    model_id = resp.data[0].id

    def ask(prompt: str) -> tuple[str, str]:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return prompt, completion.choices[0].message.content

    # Submit all prompts at once and print them in the original order
    with ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
        results = list(pool.map(ask, PROMPTS))

    print(f"Connected to {HOST}")
    for prompt, answer in results:
        print("-" * 60)
        print(f"Prompt: {prompt!r}")
        print("Output:")
        print((answer or "").strip())
    print("-" * 60)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
