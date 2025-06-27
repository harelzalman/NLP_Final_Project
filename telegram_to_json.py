import pandas as pd
import json
from pathlib import Path

# 1. Read your CSV (no header: message then label)
df = pd.read_csv(
    r"CyberMethods\Telegram_spam\datasets\telegram_train.csv",
    header=None,
    names=["message", "label"],
    dtype=str,
)

# 2. Map your text labels → desired outputs
LABEL_MAP = {
    "spam": "suspicious",
    "ham":  "non-suspicious",
}

BASE_PROMPT = (
    "Decide whether the following Telegram message is suspicious or non-suspicious."
)

out_path = Path(
    "text-generation-webui/user_data/training/datasets/telegram_train.json"
)
out_path.parent.mkdir(parents=True, exist_ok=True)

# 3. Build records
records = []
for _, row in df.iterrows():
    lbl = LABEL_MAP.get(row["label"].strip().lower())
    if lbl is None:
        continue
    records.append({
        "instruction": BASE_PROMPT,
        "input":       row["message"].strip(),
        "output":      lbl,
    })

# 4. Write out as a JSON array
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(records)} records → {out_path}")
