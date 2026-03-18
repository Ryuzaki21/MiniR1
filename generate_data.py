# i'm using groq api to generate reasoning traces for gsm8k dataset
# basically using a big model (70b) to teach a small model how to think
# run this first before training

import json
import os
import time
from groq import Groq
from datasets import load_dataset

client = Groq(api_key="YOUR_GROQ_API_KEY")

# loading gsm8k - standard math dataset
dataset = load_dataset("gsm8k", "main", split="train")

def generate_reasoning(question):
    # asking groq to solve with step by step thinking
    # tried different prompts, this one works best
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""Solve this math problem step by step.

Format your response EXACTLY like this:
<thinking>
step by step reasoning here
</thinking>
<answer>
final answer here
</answer>

Problem: {question}"""
            }
        ]
    )
    return response.choices[0].message.content

# resuming from where we left off if dataset already exists
# groq has daily rate limits so sometimes need to run this multiple times
if os.path.exists("reasoning_dataset.json"):
    with open("reasoning_dataset.json", "r") as f:
        reasoning_dataset = json.load(f)
    start_index = len(reasoning_dataset)
    print(f"found existing dataset with {start_index} samples, continuing...")
else:
    reasoning_dataset = []
    start_index = 0
    print("starting fresh")

for i, sample in enumerate(dataset.select(range(start_index, 2000))):
    try:
        question = sample["question"]
        reasoning = generate_reasoning(question)
        
        reasoning_dataset.append({
            "question": question,
            "reasoning": reasoning
        })

        # saving every 10 samples so we dont lose progress if it crashes
        if i % 10 == 0:
            with open("reasoning_dataset.json", "w") as f:
                json.dump(reasoning_dataset, f)
            print(f"saved {len(reasoning_dataset)} samples so far...")

        time.sleep(1)  # groq rate limit

    except Exception as e:
        print(f"hit rate limit at {len(reasoning_dataset)} samples")
        print(f"error: {e}")
        # save whatever we have before stopping
        with open("reasoning_dataset.json", "w") as f:
            json.dump(reasoning_dataset, f)
        break

print(f"done! total samples: {len(reasoning_dataset)}")