#!/usr/bin/env python
# vLLM client chat interface connecting to the OpenAI-compatible API server.
#
# Usage:
# Once you have a vLLM server running, set the VLLM_HOST environment variable to the server's hostname and port.
# Then launch this script with,
# ./vllm_client_chat.py

import os
from openai import OpenAI

HOST = os.environ.get("VLLM_HOST", "http://127.0.0.1:8000")
API_KEY = os.environ.get("VLLM_API_KEY", "your-secret-key")

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
    display_model = model_name or model_id

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for an HPC user.",
        }
    ]

    print(f"Connected to {HOST}")
    print(f"vLLM model: {display_model}")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            return 0

        messages.append({"role": "user", "content": user})

        print("Model:", end=" ", flush=True)
        assistant_text = ""

        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    assistant_text += delta
            print("\n")
        except Exception as e:
            print(f"\nRequest failed: {e}\n")
            continue

        messages.append({"role": "assistant", "content": assistant_text})

if __name__ == "__main__":
    raise SystemExit(main())
