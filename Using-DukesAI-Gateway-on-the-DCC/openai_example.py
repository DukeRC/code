#!/usr/bin/env python
#
# A Python script using the OpenAI-compatible API on Duke's AI Gateway to generate a response based on an input prompt.
# Courtesy of Drew Stinnett:https://ai.colab.duke.edu/colab-ai-blog/all-blogs/getting-started-with-dukes-ai-gateway-a-developers-guide
#
# Usage:
# Get your API token from: https://dashboard.ai.duke.edu/api-keys
# Set your API token in a .env file in this directory with the following content:
# LITELLM_TOKEN="your_api_token_here"
# or export the environment variable directly in your shell. Then run,
# ./openai_example.py "Your prompt here"

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Configuration
MODEL = "gpt-5.4"
INSTRUCTIONS = "You are a helpful assistant here to demo the power of AI."


def main():
    # Local .env file content
    load_dotenv()

    # Input arguments
    if len(sys.argv) < 2:
        print("Usage: ./openai_example.py <prompt>")
        sys.exit(1)
    token = os.getenv("LITELLM_TOKEN")
    if not token:
        print("Please set the LITELLM_TOKEN environment variable.")
        sys.exit(1)
    prompt = sys.argv[1]

    # Connect to the OpenAI API
    client = OpenAI(
        api_key=token,
        base_url="https://litellm.oit.duke.edu/v1",
    )

    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=prompt,
    )

    print(response.output[0].content[0].text)


if __name__ == "__main__":
    main()
