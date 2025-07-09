# NLP_Final_Project - Prompt Impact

> **Assessing how prompt design and lightweight LoRA tuning affect LLM accuracy on cyber-security classification tasks**

---

## Overview
The project explores whether careful prompt engineering alone can rival small-footprint fine-tuning for three classic security problems:

| Task               | Dataset (size)              | Labels                       |
|--------------------|-----------------------------|------------------------------|
| URL Phishing       | Kaggle 2023 — 10 k URLs     | `legitimate`, `phish`        |
| Telegram Spam      | Kaggle 2024 — 8 k messages  | `ham`, `spam`                |
| Password Strength  | Kaggle 2024 — 5 k passwords | `weak`, `medium`, `strong`   |

Four prompt families are compared:

1. **No prompt** – empty instruction  
2. **Generic** – one-line task description  
3. **Specific** – generic line + short domain cue  
4. **LoRA-tuned** – generic line + 90-MB adapter trained on 2 k labelled rows  

Key insight: a single extra cue can trade off *FPR* and *FNR* without hurting F1; LoRA offers the best balanced performance for ~7 h training on an RTX 3060.
