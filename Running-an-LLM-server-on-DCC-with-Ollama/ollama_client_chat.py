#!/usr/bin/env python

from ollama import Client

HOST = "http://dcc-h200-gpu-06.rc.duke.edu:11434"
MODEL = "llama4:scout"


def main():
    client = Client(host=HOST)

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

    # Start the conversation with an optional system prompt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for an HPC user.",
        }
    ]

    print(f"Connected to {HOST} using model {MODEL}")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        # Add user message to the conversation
        messages.append({"role": "user", "content": user})

        print("Model:", end=" ", flush=True)
        assistant_text = ""

        # Stream the response token by token
        for chunk in client.chat(
            model=MODEL,
            messages=messages,
            stream=True,
        ):
            if chunk.message.content:
                print(chunk.message.content, end="", flush=True)
                assistant_text += chunk.message.content
        print("\n")

        # Add assistant reply to history so the model remembers context
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
