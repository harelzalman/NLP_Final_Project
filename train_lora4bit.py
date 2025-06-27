"""
train_lora4bit.py  –  4-bit QLoRA fine-tuning for Meta-Llama-3-8B-Instruct
---------------------------------------------------------------------------
• Dataset : JSON-Lines file  phish_urls_train.json   (1 000 rows)
• Prompt  : "instruction \\n URL: ... \\n Answer: ..."
• Tokenise now → PyTorch tensors (so Trainer never touches Arrow)
• Adapter : r=16, α=32 on q/v/o projections   (11 M trainable params)
• Runs 10 epochs, LR 1e-4, effective batch 16  (4×4, grad-acc 4)
"""

import os, json, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ───── paths ──────────────────────────────────────────────────────────
BASE_MODEL = r"text-generation-webui\user_data\models\meta-llama3-8b-instruct"
DATA_FILE  = r"text-generation-webui\user_data\training\datasets\phish_urls_train.json"
OUT_DIR    = r"text-generation-webui\user_data\loras\NOUI\lora-phish-4bit"

# ───── tokenizer ─────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tok.pad_token = tok.eos_token

# ───── 1) build prompt strings ───────────────────────────────────────
def to_prompt(row):
    return {
        "text": (
            f"{row['instruction'].strip()}\n"
            f"URL: {row['input'].strip()}\n"
            f"Answer: {row['output'].strip()}{tok.eos_token}"
        )
    }

ds = (
    load_dataset("json", data_files=DATA_FILE, split="train")
    .map(to_prompt, remove_columns=["instruction", "input", "output"])
)

# ───── 2) tokenise + convert to tensors  (no Arrow at runtime) ───────
def tokenize(batch):
    return tok(batch["text"],
               truncation=True,
               padding="max_length",
               max_length=256)

ds = ds.map(tokenize, batched=True, remove_columns=["text"])
ds.set_format(type="torch")          # DataLoader now sees real tensors

print("Dataset ready:", len(ds), "examples")

# ───── 3) load base model in 4-bit (Bits-and-Bytes) ──────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_cfg,
    attn_implementation="eager",   # Flash-Attn OFF
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False          #  silences the Trainer warning

# ───── 4) attach QLoRA adapter ───────────────────────────────────────
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    peft_type="QLORA",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()   # should show 11 M trainables

# ───── 5) training setup ─────────────────────────────────────────────
args = TrainingArguments(
    output_dir                = OUT_DIR,
    per_device_train_batch_size=4,      # 4×4-bit ≈ 9-10 GB
    gradient_accumulation_steps=4,      # eff. batch 16
    num_train_epochs          = 10,
    learning_rate             = 1e-4,
    lr_scheduler_type         = "constant_with_warmup",
    warmup_steps              = 10,
    logging_steps             = 20,
    save_strategy             = "epoch",
    fp16                      = True,
)

data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = ds,
    data_collator   = data_collator,
)

# ───── 6) train ──────────────────────────────────────────────────────
trainer.train()

# ───── 7) save ───────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"\n  LoRA adapter saved to: {OUT_DIR}")
