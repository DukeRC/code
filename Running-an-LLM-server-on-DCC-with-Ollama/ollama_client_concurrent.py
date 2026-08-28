#!/usr/bin/env python

from concurrent.futures import ThreadPoolExecutor
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

def ask(prompt):
    resp = client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return prompt, resp.message.content

# Submit all prompts at once and print them in the original order
print("-" * 60)
print(f"Model: {MODEL}")
with ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
    for prompt, answer in pool.map(ask, PROMPTS):
        print("-" * 60)
        print(f"Prompt: {prompt!r}")
        print("Output:")
        print((answer or "").strip())
print("-" * 60)
