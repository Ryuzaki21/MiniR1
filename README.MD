# qlora-llama3.2-reasoning(MiniR1)

A small language model fine-tuned to reason through math problems step by step.

## Overview

Llama 3.2 3B fine-tuned using SFT + QLoRA on 1203 reasoning traces generated from GSM8K. The model outputs a `<thinking>` block before giving a final `<answer>`.

## Demo

[Live Demo](https://huggingface.co/spaces/Ryuzaki23/MiniR1) | [Model](https://huggingface.co/Ryuzaki23/MiniR1)

## Tech Stack

| Component | Details |
|-----------|---------|
| Base Model | meta-llama/Llama-3.2-3B |
| Training | SFT + QLoRA, 4-bit quantization, r=8 |
| Dataset | GSM8K + llama-3.3-70b-versatile via Groq API |
| Tools | transformers, peft, trl, bitsandbytes |

## Project Structure
```
MiniR1/
├── generate_data.py        # dataset generation via Groq API
├── train.py                # QLoRA fine-tuning
└── reasoning_dataset.json  # 1203 reasoning samples
```

## Getting Started
```bash
git clone git clone https://github.com/Ryuzaki21/MiniR1.git
cd MiniR1
pip install -r requirements.txt
```

## How it works

1. Generated 1203 reasoning traces from GSM8K using `llama-3.3-70b-versatile` via Groq API
2. Fine-tuned Llama 3.2 3B on Kaggle T4 GPU using QLoRA
3. Only 0.07% of parameters trained

## License

MIT
