import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle  # only for final shuffling

SEED = 42
SRC  = Path(r"CyberMethods\phishing_site_urls\datasets\phishing_site_urls.csv")

# 1. Load (with header) and ensure label is integer
df = pd.read_csv(SRC)
df["label"] = df["label"].astype(int)

# 2. Sample per class (no overlap)
splits = {"train": [], "test": []}
for cls, n_train, n_test in [(1, 500, 50), (0, 500, 50)]:
    pool = df[df["label"] == cls].sample(n_train + n_test, random_state=SEED)
    splits["train"].append(pool.iloc[:n_train])
    splits["test"].append(pool.iloc[n_train:])

train_df = shuffle(pd.concat(splits["train"]), random_state=SEED)
test_df  = shuffle(pd.concat(splits["test"]),  random_state=SEED)

# 3. Write out (no header)
OUT_DIR    = Path("datasets/Twitter")
train_path = OUT_DIR / "twitter_train.csv"
test_path  = OUT_DIR / "twitter_test.csv"

train_df.to_csv(train_path, index=False, header=False)
test_df.to_csv(test_path,  index=False, header=False)

print(f"✓ Wrote {len(train_df)} rows → {train_path}")
print(f"✓ Wrote {len(test_df)} rows → {test_path}")
