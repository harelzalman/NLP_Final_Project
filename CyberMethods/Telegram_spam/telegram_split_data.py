import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

SEED = 42
SRC  = Path(r"datasets\Telegram_Spam_or_Ham.csv")

# 1. Load without headers, name the columns text and label
df = pd.read_csv(SRC)

# 2. Sample per class (no overlap)
#    We want 1 000 train (500 ham, 500 spam) and 250 test (125 ham, 125 spam)
splits = {"train": [], "test": []}
for cls, n_train, n_test in [("ham", 500, 125), ("spam", 500, 125)]:
    pool = df[df["text_type"] == cls].sample(n_train + n_test, random_state=SEED)
    splits["train"].append(pool.iloc[:n_train])
    splits["test"].append(pool.iloc[n_train:])

train_df = shuffle(pd.concat(splits["train"]), random_state=SEED)
test_df  = shuffle(pd.concat(splits["test"]),  random_state=SEED)

# 3. Write out with no header
OUT_DIR = Path("datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUT_DIR / "telegram_train.csv", index=False, header=False)
test_df.to_csv(OUT_DIR / "telegram_test.csv",  index=False, header=False)

print(f"✓ Wrote {len(train_df)} rows → {OUT_DIR / 'telegram_train.csv'}")
print(f"✓ Wrote {len(test_df)} rows → {OUT_DIR / 'telegram_test.csv'}")
