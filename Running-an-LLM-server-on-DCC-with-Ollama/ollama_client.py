#!/usr/bin/env python

from ollama import Client
import os

MODEL = "qwen3.6:latest"

# Array of prompts
PROMPTS = [
    "Tell me about Sri Lanka",
    "Why is the sky blue?",
    "Write a Python code that calculates the Fibonacci sequence up to 15.",
]

client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

# Check if model already exists
resp = client.list()
models = resp.models
model_names = [m.model for m in models]

if MODEL not in model_names:
    try:
        print(f"Pulling model '{MODEL}'...")
        client.pull(model=MODEL)
    except Exception as e:
        print(f"Could not pull model: {e}")

# Run each prompt in turn, streaming output token by token
print("-" * 60)
print(f"Model: {MODEL}")
for i, prompt in enumerate(PROMPTS, start=1):
    print("-" * 60)
    print(f"Prompt {i}/{len(PROMPTS)}: {prompt!r}")
    print("Output:")
    for chunk in client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        print(chunk.message.content, end="", flush=True)
    print()
print("-" * 60)
