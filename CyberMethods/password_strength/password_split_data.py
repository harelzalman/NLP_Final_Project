
import csv
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


SEED = 42
SRC  = Path(r"datasets\password_strength.csv")

# 1. Read every line with csv.reader (handles quoted commas)
records = []
with open(SRC, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)          # skip header row
    for row in reader:
        if len(row) < 2:
            continue
        # assume the last field is the label ("good"/"bad")
        password, label = row[-2], row[-1]
        records.append((password, label))

df = pd.DataFrame(records, columns=["password", "label"])
df["label"] = df["label"].astype(str)

# 2. Stratified split → exactly 1 000 train / 250 test
train_df, test_df = train_test_split(
    df,
    train_size=1000,
    test_size=250,
    stratify=df["label"],
    random_state=SEED,
)


# 3. (Optional) Map labels if you’d rather use words
label_map = {
    "0": "weak",  # weak
    "1": "medium",  # medium
    "2": "strong",  # strong
}
train_df["label"] = train_df["label"].map(label_map)
test_df["label"]  = test_df["label"].map(label_map)


# 4. Write out with no header
OUT_DIR = Path("datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUT_DIR / "password_train.csv", index=False, header=False)
test_df.to_csv(OUT_DIR / "password_test.csv",  index=False, header=False)

print(f"✓ Wrote {len(train_df)} rows → {OUT_DIR / 'password_train.csv'}")

# 4. Print counts per label in each split
print("Train split counts:")
print(train_df["label"].value_counts(), "\n")


print(f"✓ Wrote {len(test_df)} rows → {OUT_DIR / 'password_test.csv'}")


print("Test split counts:")
print(test_df["label"].value_counts())