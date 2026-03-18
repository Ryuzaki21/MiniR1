# fine tuning llama 3.2 3b on our reasoning dataset
# using qlora because i dont have a $10k gpu lol
# trained this on kaggle t4 gpu (free)

import os
import json
import torch
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# single gpu works better for small models
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

login(token="YOUR_HF_TOKEN")

# loading the dataset i generated with generate_data.py
with open("reasoning_dataset.json", "r") as f:
    loaded_data = json.load(f)
print(f"loaded {len(loaded_data)} samples")

# llama 3.2 has its own special format, this took me a while to figure out
def format_sample(sample):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{sample['question']}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{sample['reasoning']}
<|eot_id|>"""

hf_dataset = Dataset.from_dict({"text": [format_sample(s) for s in loaded_data]})
print(f"formatted {len(hf_dataset)} samples ready for training")

# 4bit quantization - reduces model from ~6gb to ~2gb
# cant train without this on free gpu
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

print("loading model... this takes a few minutes")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

# preparing model for lora training
model = prepare_model_for_kbit_training(model)

# lora config - only training 0.07% of parameters
# r=8 worked well, tried 16 but no significant improvement
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# batch size 2 because t4 runs out of memory with anything higher
# gradient accumulation makes effective batch size = 2 * 8 = 16
training_args = TrainingArguments(
    output_dir="./minir1",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    report_to="none",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    dataloader_pin_memory=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset
)

print("starting training...")
trainer.train()

# saving locally and pushing to huggingface
model.save_pretrained("./minir1-model")
tokenizer.save_pretrained("./minir1-model")
print("model saved locally!")

model.push_to_hub("Ryuzaki23/MiniR1")
tokenizer.push_to_hub("Ryuzaki23/MiniR1")
print("pushed to huggingface!")