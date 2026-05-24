#!/usr/bin/env python

from ollama import Client
import os

MODEL = "llama4:scout"
PROMPT = "Write a Python code that calculates the Fibonacci sequence up to 15."

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

# Stream output token by token
for chunk in client.chat(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    stream=True,
):
    print(chunk.message.content, end="", flush=True)
print("\n")
