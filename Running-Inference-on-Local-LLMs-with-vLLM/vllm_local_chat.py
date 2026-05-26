#!/usr/bin/env python
#
# An example of using vLLM with a local model for interactive chat.
#
# Usage:
#   1. Set HF_HOME to control where Hugging Face cache is stored.
#      export HF_HOME=/path/to/hf_cache
#   2. Set the HF_TOKEN environmental variable in a .env file in the root of the directory.
#   3. Run the script:
#      ./vllm_local_chat.py

import os
from dotenv import load_dotenv

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
from vllm import LLM, SamplingParams

# Model configuration
MODEL = "Qwen/Qwen2-7B"
SAMPLING_PARAMS = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=256)
SYSTEM_PROMPT = "You are a helpful assistant."

def main() -> None:
    # Connect to Hugging Face with HF_TOKEN from .env
    load_dotenv()

    llm = LLM(model=MODEL, trust_remote_code=True)
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Model: {MODEL}")
    print(f"Enter {sorted({"exit", "quit", "q"})} to exit.")
    print("-" * 60)

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit", "q"}:
            print("Exiting chat.")
            break

        conversation.append({"role": "user", "content": user_text})
        outputs = llm.chat(
            conversation, sampling_params=SAMPLING_PARAMS, use_tqdm=False
        )
        assistant_text = outputs[0].outputs[0].text.strip()
        print(f"Assistant: {assistant_text}\n")
        conversation.append({"role": "assistant", "content": assistant_text})

if __name__ == "__main__":
    main()
