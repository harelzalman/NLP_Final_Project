import pandas as pd
import json

# 1. Read your CSV
df = pd.read_csv(
    r"datasets\urls_train.csv",
    header=None,
    names=["url","label"],
    dtype=str,
)

# 2. Map your labels → model outputs
LABEL_MAP = {"bad": "phish", "good": "legitimate"}

BASE_PROMPT = (
    "Decide whether the following URL is legitimate or phish."
)

out_path = "../../text-generation-webui/user_data/training/datasets/urls_train.json"

with open(out_path, "w", encoding="utf-8") as f:
    for i, row in df.iterrows():
        lbl = LABEL_MAP.get(row["label"].strip().lower())
        if lbl is None:
            continue
        rec = {
            "instruction": BASE_PROMPT,
            "input":       row["url"].strip(),
            "output":      lbl,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote dataset to {out_path} ({len(df)} rows processed)")
