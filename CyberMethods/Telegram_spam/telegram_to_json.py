import pandas as pd
import json
from pathlib import Path

# 1. Read with no header, naming columns explicitly
df = pd.read_csv(
    r"datasets/telegram_train.csv",
    header=None,
    names=["message", "label"],
    dtype=str
)

BASE_PROMPT = (
    "Decide whether the following Telegram message is 'spam' or 'ham'."
)

prompt = (
    "### SYSTEM ###\n"
    f"{BASE_PROMPT}\n"
    "Respond with exactly ONE word 'spam' or 'ham' after ### ANSWER ###, no other tokens.\n\n"

)


out_path = Path(
    "../../text-generation-webui/user_data/training/datasets/telegram_train.json"
)
out_path.parent.mkdir(parents=True, exist_ok=True)

# 2. Build records
with open(out_path, "w", encoding="utf-8") as f:
    for i, row in df.iterrows():
        lbl = row["label"].strip().lower()
        if lbl is None:
            continue

        INPUT = ("### MESSAGE ###\n\n"
                 f"{row['message'].strip()}\n\n"
                 "### ANSWER ###\n\n")
        rec = {
            "instruction": prompt,
            "input":       INPUT,
            "output":      lbl,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote dataset to {out_path} ({len(df)} rows processed)")

