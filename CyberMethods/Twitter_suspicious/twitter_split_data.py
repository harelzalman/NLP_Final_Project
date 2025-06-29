import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

SEED = 42
SRC  = Path(r"datasets\suspicious_tweets.csv")

# 1. Load without headers, name the columns text and label
df = pd.read_csv(SRC)
df['label'] = df['label'].astype(str)


# 2. Sample per class (no overlap)
#    We want 1 000 train (500 1-non_suspicious, 500 0-suspicious) and 250 test (125 1-non_suspicious, 125 0-suspicious)
splits = {"train": [], "test": []}
for cls, n_train, n_test in [("0", 500, 125), ("1", 500, 125)]:
    pool = df[df["label"] == cls].sample(n_train + n_test, random_state=SEED)
    splits["train"].append(pool.iloc[:n_train])
    splits["test"].append(pool.iloc[n_train:])


train_df = shuffle(pd.concat(splits["train"]), random_state=SEED)
test_df  = shuffle(pd.concat(splits["test"]),  random_state=SEED)


# 3. Map numeric labels → descriptive
label_map = {
    "0": "suspicious",
    "1": "non-suspicious",
}

train_df["label"] = train_df["label"].map(label_map)
test_df["label"]  = test_df["label"].map(label_map)

# 4. Write out with no header
OUT_DIR = Path("datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUT_DIR / "twitter_train.csv", index=False, header=False)
test_df.to_csv(OUT_DIR / "twitter_test.csv",  index=False, header=False)

print(f"✓ Wrote {len(train_df)} rows → {OUT_DIR / 'twitter_train.csv'}")
print(f"✓ Wrote {len(test_df)} rows → {OUT_DIR / 'twitter_test.csv'}")
