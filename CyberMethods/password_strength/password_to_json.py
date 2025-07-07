import pandas as pd
import json
from pathlib import Path

# 1. Read your CSV (no header, message then label)
df = pd.read_csv(
    r"datasets\password_train.csv",
    header=None,
    names=["password","label"],
    dtype=str,
)


BASE_PROMPT = (
    "Decide whether the following password is 'weak', 'medium', or 'strong'."
)

prompt = (
     "### SYSTEM ###\n"
     f"{BASE_PROMPT}\n"
     "Respond with exactly ONE word 'weak', 'medium', or 'strong' after ### ANSWER ###, no other tokens.\n\n"
)



out_path = Path(
    "../../text-generation-webui/user_data/training/datasets/password_train.json"
)
out_path.parent.mkdir(parents=True, exist_ok=True)

# 2. Build records
with open(out_path, "w", encoding="utf-8") as f:
    for i, row in df.iterrows():
        lbl = row["label"].strip().lower()
        if lbl is None:
            continue
        #lbl = 1 if lbl == "suspicious" else 0
        INPUT = ("### PASSWORD ###\n\n"
                 f"{row['password'].strip()}\n\n"
                 "### ANSWER ###\n\n")

        rec = {
            "instruction": prompt,
            "input":       INPUT,
            "output":      lbl,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote dataset to {out_path} ({len(df)} rows processed)")

