#!/usr/bin/env python
#
# A Python script using the OpenAI-compatible API on Duke's AI Gateway to generate responses for prompts in a CSV file.
#
# Usage:
# Set your API token in a .env file in this directory with the following content:
# LITELLM_TOKEN="your_api_token_here"
# The input CSV must have "id" and "prompt" columns. Then run,
# ./openai_csv.py prompts.csv responses.csv

import csv
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
    if len(sys.argv) < 3:
        print("Usage: ./openai_csv.py <input.csv> <output.csv>")
        sys.exit(1)
    token = os.getenv("LITELLM_TOKEN")
    if not token:
        print("Please set the LITELLM_TOKEN environment variable.")
        sys.exit(1)
    infile, outfile = sys.argv[1], sys.argv[2]

    # Connect to the OpenAI API
    client = OpenAI(
        api_key=token,
        base_url="https://litellm.oit.duke.edu/v1",
    )

    with open(infile, newline="") as fin, open(outfile, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["id", "prompt", "response"])
        writer.writeheader()

        for row in reader:
            # Record failures and keep going so one bad request doesn't end the loop
            try:
                response = client.responses.create(
                    model=MODEL,
                    instructions=INSTRUCTIONS,
                    input=row["prompt"],
                )
                text = response.output[0].content[0].text
            except Exception as e:
                text = f"ERROR: {e}"

            writer.writerow({"id": row["id"], "prompt": row["prompt"], "response": text})
            fout.flush()
            print(f"Finished prompt {row['id']}")


if __name__ == "__main__":
    main()
