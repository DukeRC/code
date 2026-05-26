#!/usr/bin/env python
#
# An example of using vLLM with a local model.
#
# Usage:
#   1. Set HF_HOME to control where Hugging Face cache is stored.
#      export HF_HOME=/path/to/hf_cache
#   2. Set the HF_TOKEN environmental variable in a .env file in the root of the directory.
#   3. Run the script:
#      ./vllm_local.py

import os
from dotenv import load_dotenv

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
from vllm import LLM, SamplingParams

# Model configuration
MODEL = "Qwen/Qwen2-7B"
SAMPLING_PARAMS = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=256)

# Array of prompts
PROMPTS = ["Tell me about Sri Lanka", "Why is the sky blue?", "Write a Python code that calculates the Fibonacci sequence up to 15."]

def main():
    # Connect to Hugging Face with HF_TOKEN from .env
    load_dotenv()

    # Launch LLM
    llm = LLM(model=MODEL, trust_remote_code=True)
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)

    print("-" * 60)
    print(f"Model: {MODEL}")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print("Output:")
        print(generated_text.strip())
        print("-" * 60)

if __name__ == "__main__":
    main()
