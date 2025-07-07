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
CSV_PATH    = Path(r"datasets/telegram_test.csv")
#OUTPUT_DIR  = Path(r"results")
OUTPUT_DIR  = Path(r"results/with_lora")
NUM_TRIES   = 1
LLAMA3_URL  = "http://localhost:5000/v1/completions"
MODEL_NAME  = "llama-3-8b-instruct"

# two system prompts: first empty (no-prompt baseline), second our generic Telegram prompt
BASE_PROMPT = (

    # 0 No Prompt
    "",
    # 1 Generic Prompt
    "Decide whether the following Telegram message is 'spam' or 'ham'.",
    # 2 Lexical & Syntax Patterns
    # "Examine the Telegram message’s lexical and syntactic patterns, paying attention to unusual punctuation (e.g., \"!!!\"), excessive capitalization, and repetitive phrasing. Compare these signals against typical conversational norms—if the structure deviates sharply from personal chat style, label it as spam; otherwise, mark as ham.",
    # # 3 URL & Domain Reputation
    # "Evaluate the message for embedded URLs, shortened links, or attachments. For any links found, cross-reference their domains against known safe versus risky lists. If the message contains high-risk or obfuscated links, classify it as spam; if links are from trusted domains or none are present, classify as ham.",
    # # 4 Tone & Urgency Detection
    # "Perform sentiment and urgency analysis: spam often uses highly promotional language (\"limited time,\" \"act now\") or creates false urgency. If the message tone is pushy, overly sales-driven, or fear-inducing, mark as spam; if it reads like a calm, personal exchange, mark as ham.",
    # # 5 Call-to-Action Keywords
    # "Scan for solicitation or call-to-action phrases such as \"click here,\" \"buy,\" \"subscribe,\" \"win,\" or \"free.\" Presence of these marketing triggers should push the classification toward spam; absence of such explicit prompts suggests ham.",
    # # 6 Phishing & Social-Engineering Cues
    # "Assess the message for social-engineering elements—impersonation of brands, requests for credentials, or pleas for personal information. If any phishing indicators appear, tag as spam; if the message stays within normal conversational bounds, tag as ham.",
    # # 7 Contextual Personalization
    # "Check for personalization markers: direct references to prior conversation, recipient name, or group-specific context indicate genuine ham. Generic or broadcast-style content without context (e.g., \"Hello user, click…\") should be classified as spam.",
    # # 8 Formatting Anomalies & Character Analysis
    # "Compute metrics like the ratio of unique versus total characters, presence of long runs of emojis or non-alphabet symbols, and excessive whitespace. High anomaly scores (e.g., \"!!!!!😂😂😂\") often signal automated spam; more natural formatting implies ham.",
    # # 9 Topic Modeling
    # "Apply a lightweight topic model or keyword clustering to identify if the message focuses on marketing or sales themes versus everyday chatter. If content aligns with promotions or affiliate-style language, label as spam; if centering on personal updates or social talk, label as ham.",
    # # 10 Sender Behavior & Metadata
    # "Incorporate available metadata: frequency of identical messages sent, rapid-fire broadcasting, or sender reputation. Messages matching mass-send patterns or high send rates should be flagged as spam; one-off, varied messages lean toward ham.",
    #
    # # — Now 5 prompts to reduce false negatives (catch more spam) —
    # # 11
    # "Decide whether the following Telegram message is 'spam' or 'ham'. If unsure, label it as spam to avoid missing harmful content.",
    # # 12
    # "Decide whether the following Telegram message is 'spam' or 'ham'. Mark as spam any message containing unsolicited promotions or suspicious links.",
    # # 13
    # "Decide whether the following Telegram message is 'spam' or 'ham'. When in doubt, lean toward 'spam' to minimize false negatives.",
    # # 14
    # "Decide whether the following Telegram message is 'spam' or 'ham'. Flag borderline cases as spam to ensure no spam slips through.",
    # # 15
    # "Decide whether the following Telegram message is 'spam' or 'ham'. Prefer 'spam' if the message appears unsolicited or disruptive.",
    # # — Now 5 prompts to reduce false positives (avoid over-labeling) —
    # # 16
    # "Decide whether the following Telegram message is 'spam' or 'ham'. Only label as spam if clearly unsolicited or harmful to reduce false positives.",
    # # 17
    # "Decide whether the following Telegram message is 'spam' or 'ham'. Avoid labeling conversational or personal content as spam unless explicitly promotional.",
    # # 18
    # "Decide whether the following Telegram message is 'spam' or 'ham'. Be conservative: only tag messages with clear spam indicators as spam.",
    # # 19
    # "Decide whether the following Telegram message is 'spam' or 'ham'. Do not mark messages as spam solely on length or urgency; require explicit spam cues.",
    # # 20
    # "Decide whether the following Telegram message is 'spam' or 'ham'. Ensure genuine personal or group conversation is labeled 'ham' to minimize false positives."

)
SYS_PROMPTS = BASE_PROMPT

# ─── LOGGING ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ─── HELPERS ─────────────────────────────────────────────────────────

def normalise_gt(label: str) -> str:
    l = label.strip().lower()
    return "spam" if l == "spam" else "ham"

async def call_llama3_async(session: aiohttp.ClientSession, prompt: str,
                            max_tokens: int = 1,
                            temperature: float = 0.0) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with session.post(LLAMA3_URL, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    # adapt to response format
    if "text" in data:
        return data["text"]
    if "choices" in data and data["choices"]:
        return data["choices"][0].get("text", "")
    return ""

async def classify_message_async(session: aiohttp.ClientSession,
                                 message: str,
                                 system_text: str
                                 ) -> Tuple[str, str]:
    prompt = (
        "### SYSTEM ###\n"
        f"{system_text}\n"
        "Respond with exactly ONE word 'spam' or 'ham' after ### ANSWER ###, no other tokens.\n\n"
        "### MESSAGE ###\n\n"
        f"{message}\n\n"
        "### ANSWER ###\n\n"
    )
    raw_out = await call_llama3_async(session, prompt)
    # try to extract the answer
    m = re.search(r"###\s*ANSWER\s*###\s*(spam|ham)", raw_out, re.IGNORECASE)
    if m:
        lab = m.group(1).lower()
    else:
        m2 = re.search(r"\b(spam|ham)\b", raw_out, re.IGNORECASE)
        lab = m2.group(1).lower() if m2 else "unknown"
    return (lab, raw_out)

def plot_confusion_matrix(tp: int, fp: int, tn: int, fn: int, save_path: Path) -> None:
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Pred ham", "Pred spam"])
    ax.set_yticks([0, 1], labels=["GT ham", "GT spam"])
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ─── MAIN ────────────────────────────────────────────────────────────

async def main():
    logging.info(f"Reading CSV from {CSV_PATH}")
    rows: List[Tuple[str, str]] = []
    # read CSV: two columns message,label
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # if header is actual column names, skip it
        if header and header[0].lower() == "message":
            pass
        else:
            # rewind if first row was data
            f.seek(0)
            reader = csv.reader(f)
        for idx, row in enumerate(reader, 1):
            if len(row) < 2:
                logging.warning(f"Skipping malformed row {idx}: {row}")
                continue
            msg, lbl = row[0].strip(), row[1].strip()
            rows.append((msg, normalise_gt(lbl)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    async with aiohttp.ClientSession() as session:
        for idx, system_text in enumerate(SYS_PROMPTS, start=1):
            tp = fp = tn = fn = skipped = 0
            per_msg_results: List[Dict[str, Any]] = []

            summary_path = OUTPUT_DIR / f"llm_results_summary_{idx}.json"
            cm_path      = OUTPUT_DIR / f"llm_confusion_matrix_{idx}.png"

            print(f"\n=== Prompt #{idx} ===")
            for i, (msg, gt) in enumerate(rows, 1):
                print(f"[{i}/{len(rows)}] {msg[:50]}...")
                trials = []
                for t in range(NUM_TRIES):
                    pred, raw = await classify_message_async(session, msg, system_text)
                    trials.append({"raw": raw, "extracted": pred})
                    short = raw.replace("\n", " ")[:100] + ("..." if len(raw) > 100 else "")
                    print(f"  Trial {t+1}: extracted={pred} | raw=\"{short}\"")

                # majority vote (or first non-unknown)
                final = next((d["extracted"] for d in trials if d["extracted"] != "unknown"), "unknown")
                print(f"   → Final: {final} (GT: {gt})")

                if final == "unknown":
                    skipped += 1
                else:
                    if gt == "spam" and final == "spam":
                        tp += 1
                    elif gt == "spam" and final == "ham":
                        fn += 1
                    elif gt == "ham"  and final == "ham":
                        tn += 1
                    else:
                        fp += 1

                per_msg_results.append({
                    "message": msg,
                    "ground_truth": gt,
                    "final_prediction": final,
                    "trials": trials
                })

            # save detailed summary
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall    = tp / (tp + fn) if tp + fn else 0.0
            f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            fpr       = fp / (fp + tn) if fp + tn else 0.0

            summary = {
                "prompt": system_text,
                "counts": {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "unknown": skipped},
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "num_samples": len(rows),
                "per_message_results": per_msg_results
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logging.info(f"Saved summary to {summary_path}")

            if any((tp, fp, tn, fn)):
                plot_confusion_matrix(tp, fp, tn, fn, cm_path)
                logging.info(f"Saved CM to {cm_path}")
            else:
                logging.warning("All predictions unknown; skipped plotting.")

            all_summaries.append({
                "prompt": system_text,
                "counts": summary["counts"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "num_samples": len(rows)
            })

    # combined summary
    combo_path = OUTPUT_DIR / "all_llm_summary.json"
    combo = {"runs": all_summaries}
    combo_path.write_text(json.dumps(combo, indent=2), encoding="utf-8")
    logging.info(f"Wrote combined summary to {combo_path}")


if __name__ == "__main__":
    asyncio.run(main())
