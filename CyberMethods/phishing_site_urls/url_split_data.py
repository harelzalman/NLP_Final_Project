import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

SEED = 42
SRC  = Path(r"datasets\phishing_site_urls.csv")

# 1. Load without headers, name the columns url and label
df = pd.read_csv(SRC, header=None, names=["url", "label"])

# 2. Sample per class (no overlap)
#    We want 1 000 train (500 good, 500 bad) and 250 test (125 good, 125 bad)
splits = {"train": [], "test": []}
for cls, n_train, n_test in [("good", 500, 125), ("bad", 500, 125)]:
    pool = df[df["label"] == cls].sample(n_train + n_test, random_state=SEED)
    splits["train"].append(pool.iloc[:n_train])
    splits["test"].append(pool.iloc[n_train:])

train_df = shuffle(pd.concat(splits["train"]), random_state=SEED)
test_df  = shuffle(pd.concat(splits["test"]),  random_state=SEED)

# 3. Write out with no header
OUT_DIR = Path("datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUT_DIR / "urls_train.csv", index=False, header=False)
test_df.to_csv(OUT_DIR / "urls_test.csv",  index=False, header=False)

print(f"✓ Wrote {len(train_df)} rows → {OUT_DIR / 'urls_train.csv'}")
print(f"✓ Wrote {len(test_df)} rows → {OUT_DIR / 'urls_test.csv'}")
