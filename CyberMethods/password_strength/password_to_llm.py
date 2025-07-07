# password_to_llm.py

import csv
import json
import asyncio
import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

import aiohttp
import matplotlib.pyplot as plt
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────
CSV_PATH    = Path(r"datasets/password_test.csv")
OUTPUT_DIR  = Path(r"results")
#OUTPUT_DIR  = Path(r"results/with_lora")
NUM_TRIES   = 1
LLAMA3_URL  = "http://localhost:5000/v1/completions"
MODEL_NAME  = "llama-3-8b-instruct"

# ─── SYSTEM PROMPTS ──────────────────────────────────────────────────
BASE_PROMPT = (
    # 0: No prompt
    "",
    # 1: General Prompt
    "Decide whether the following password is 'weak', 'medium', or 'strong'.",
    # 2: Password Length Analysis
    "Evaluate the length of the password: passwords under 8 characters are often weak; those between 8 and 12 characters medium; above 12 characters strong.",
    # 3: Character Variety Check
    "Check how many different character types the password contains (lowercase, uppercase, digits, symbols). More types generally mean a stronger password.",
    # 4: Dictionary Word Detection
    "Identify any dictionary words or common substrings (e.g., 'password', 'admin'). Presence of such words should lower the strength rating.",
    # 5: Common Patterns & Substitutions
    "Detect common patterns or leetspeak substitutions (e.g., 'p@ssw0rd', '1234'). If these are present, label the password as weaker.",
    # 6: Repetition & Sequential Characters
    "Check for repeated characters (e.g., 'aaaa') or sequential runs (e.g., '1234', 'abcd'). Sequences and repeats reduce password strength.",
    # 7: Entropy Estimation
    "Estimate the password’s entropy based on length and character variety: higher estimated entropy indicates stronger passwords.",
    # 8: Keyboard Pattern Analysis
    "Assess whether the password follows keyboard patterns (e.g., 'qwerty', 'asdfgh'). Such patterns are easy to guess and should be marked weak.",
    # 9: Symbol & Digit Placement
    "Analyze where symbols and digits appear: random placement boosts strength; predictable placements (e.g., only at the end) reduce it.",
    # 10: Overall Complexity Summary
    "Combine all the above analyses into an overall strength rating of 'weak', 'medium', or 'strong'.",


    # Prompts to reduce FPR (avoid marking strong as weak; assume most are medium)
    "Verify high-entropy segments: if long, truly random sequences appear, bias toward 'strong' over 'medium' or 'weak', given most passwords are medium.",
    "Cross-reference against known breach corpora: if a complex password has never appeared in large breach lists, lower the chance of false weak labels, remembering that true weak cases are rarer.",
    "Recognize uncommon character distributions (e.g., base64-style strings): treat these as 'strong' even if length is borderline, because mislabeling a rare strong as weak is costly.",

    # Prompts to reduce FNR (avoid missing weak; remembering medium dominates)
    "Detect personal-info patterns (birthdates, names, phone numbers): flag as 'weak' regardless, since these guessable cases—though less frequent—must be caught.",
    "Identify short dictionary words with substitutions (e.g., 'P@ss', 'Adm1n'): even if leetified, mark 'weak', because missing these rare weak examples increases risk.",
    "Search for common date or year formats (e.g., '1987', '2025'): embedded dates drastically lower security and should always score 'weak', despite class imbalance.",

    # 18
    # Detect keyboard adjacency sequences (e.g., 'qwer', 'zxcv'): these patterns are easy to guess and should lower strength.
    "Analyze keyboard adjacency patterns: flag repeated runs along rows or columns (e.g., 'asdf', '1234') as weaker due to guessability.",

    # Check for long single‐class segments (e.g., all letters then all digits)
    "Identify large contiguous segments of a single character class (letters, digits, symbols): long homogeneous blocks reduce entropy and strength.",

    # Estimate pronounceability (i.e., looks like a real word)
    "Evaluate pronounceability by checking vowel–consonant patterns: highly pronounceable strings are often dictionary‐derived and should lean weaker.",

    # Look at bigram frequency against English language corpus
    "Score the password’s bigram/trigram frequencies against an English corpus: high‐frequency n-grams indicate predictability and lower strength.",

    # Detect palindromes or mirrored substrings
    "Detect palindromic or mirrored substrings (e.g., 'abba', '1221'): these symmetrical patterns reduce randomness and should decrease strength.",

    # All Heuristics
    "Perform a comprehensive heuristic analysis—evaluate length, character-type variety, entropy estimates, dictionary words (including leet substitutions), personal-info patterns, keyboard adjacency and sequences, homogeneous segments, common date/year formats, n-gram frequencies, pronounceability, palindromes, symbol/digit placement, and breach-list occurrences—and combine all findings into a final strength rating of 'weak', 'medium', or 'strong'."

)

SYS_PROMPTS = BASE_PROMPT

# ─── LOGGING ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ─── HELPERS ─────────────────────────────────────────────────────────
LABELS    = ["weak", "medium", "strong"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABELS)}

def normalise_gt(label: str) -> str:
    l = label.strip().lower()
    return l if l in LABELS else "unknown"

async def call_llama3_async(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int = 1,
    temperature: float = 0.0
) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with session.post(LLAMA3_URL, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    if "text" in data:
        return data["text"]
    if "choices" in data and data["choices"]:
        return data["choices"][0].get("text", "")
    return ""

async def classify_password_async(
    session: aiohttp.ClientSession,
    password: str,
    system_text: str
) -> Tuple[str, str]:
    prompt = (
        "### SYSTEM ###\n"
        f"{system_text}\n"
        "Respond with exactly ONE word 'weak', 'medium', or 'strong' after ### ANSWER ###, no other tokens.\n\n"
        "### PASSWORD ###\n\n"
        f"{password}\n\n"
        "### ANSWER ###\n\n"
    )
    raw = await call_llama3_async(session, prompt)
    m = re.search(r"(weak|medium|strong)", raw, re.IGNORECASE)
    lab = m.group(1).lower() if m else "unknown"
    return lab, raw

def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ─── MAIN ────────────────────────────────────────────────────────────
async def main():
    logging.info(f"Reading CSV from {CSV_PATH}")
    rows: List[Tuple[str, str]] = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not (header and header[0].lower() == "password"):
            f.seek(0)
            reader = csv.reader(f)
        for idx, row in enumerate(reader, 1):
            if len(row) < 2:
                logging.warning(f"Skipping malformed row {idx}: {row}")
                continue
            pwd, lbl = row[0].strip(), row[1].strip()
            rows.append((pwd, normalise_gt(lbl)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    async with aiohttp.ClientSession() as session:
        for idx, system_text in enumerate(SYS_PROMPTS, start=1):
            # initialize 3×3 confusion matrix
            cm = np.zeros((3, 3), dtype=int)
            skipped = 0
            per_msg = []

            summary_path = OUTPUT_DIR / f"llm_results_summary_{idx}.json"
            cm_path      = OUTPUT_DIR / f"llm_confusion_matrix_{idx}.png"


            print(f"=== Prompt #{idx}: {system_text!r} ===")
            for i, (pwd, gt) in enumerate(rows, 1):
                print(f"[{i}/{len(rows)}] Testing '{pwd}' (GT={gt})…")
                trials = []
                for t in range(NUM_TRIES):
                    pred, raw = await classify_password_async(session, pwd, system_text)
                    trials.append(pred)
                    print(f"  Trial {t+1}: extracted={pred} ")
                final = next((p for p in trials if p != "unknown"), trials[-1])

                # update 3×3 matrix
                gt_i   = LABEL2IDX.get(gt,   None)
                pred_i = LABEL2IDX.get(final, None)
                if gt_i is None or pred_i is None:
                    skipped += 1
                else:
                    cm[gt_i, pred_i] += 1

            # ─── METRICS (macro-average over weak, medium, strong) ────────────────
            total = int(cm.sum())
            # overall accuracy
            correct = int(np.trace(cm))
            accuracy = correct / total if total else 0.0

            # per-class stats and accumulators
            precisions, recalls, f1s, fprs, fnrs = [], [], [], [], []
            for lbl in LABELS:
                i = LABEL2IDX[lbl]
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = total - tp - fp - fn

                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) else 0.0
                fpr = fp / (fp + tn) if (fp + tn) else 0.0
                fnr = fn / (fn + tp) if (fn + tp) else 0.0

                precisions.append(p)
                recalls.append(r)
                f1s.append(f1)
                fprs.append(fpr)
                fnrs.append(fnr)

            # macro-averaged metrics
            precision = sum(precisions) / len(precisions)
            recall = sum(recalls) / len(recalls)
            f1 = sum(f1s) / len(f1s)
            fpr = sum(fprs) / len(fprs)
            fnr = sum(fnrs) / len(fnrs)


            plot_confusion_matrix(cm, cm_path)

            summary = {
                "prompt": system_text,
                "confusion_matrix": cm.tolist(),
                "num_samples": total,
                "skipped": skipped,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logging.info(f"Saved summary to {summary_path}")
            all_summaries.append(summary)

    combo_path = OUTPUT_DIR / "all_llm_summary.json"
    combo_path.write_text(json.dumps({"runs": all_summaries}, indent=2), encoding="utf-8")
    logging.info(f"Wrote combined summary to {combo_path}")

if __name__ == "__main__":
    asyncio.run(main())
