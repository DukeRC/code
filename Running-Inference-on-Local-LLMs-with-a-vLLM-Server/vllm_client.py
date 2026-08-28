#!/usr/bin/env python
# vLLM client connecting to the OpenAI-compatible API server.
#
# Usage:
# Once you have a vLLM server running, set the VLLM_HOST environment variable to the server's hostname and port.
# Then launch this script with,
# ./vllm_client.py

import os
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

    model_obj = resp.data[0]
    model_id = model_obj.id
    model_name = getattr(model_obj, "root", None)

    print(f"Connected to {HOST}")
    display_model = model_name or model_id
    print(f"vLLM model: {display_model}")
    print()

    # Send each prompt in turn, streaming the response token by token
    for i, prompt in enumerate(PROMPTS, start=1):
        print("-" * 60)
        print(f"Prompt {i}/{len(PROMPTS)}: {prompt!r}")
        print("Output:")
        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
            print()
        except Exception as e:
            print(f"Request failed: {e}")
            return 1
    print("-" * 60)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
