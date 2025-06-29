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
CSV_PATH    = Path(r"datasets\urls_test.csv")
#OUTPUT_DIR  = Path(r"results")
OUTPUT_DIR  = Path(r"results\with_lora")
NUM_TRIES   = 1
LLAMA3_URL  = "http://localhost:5000/v1/completions"
MODEL_NAME  = "llama-3-8b-instruct"
BASE_PROMPT = (

   # 0 No Prompt
   "",
   # 1 Generic Prompt
   "Decide whether the following URL is legitimate or phish."
   # # 2 Step-by-Step Heuristics Prompt
   # "Analyse the URL in four stages—host, path, query, then overall—flagging IP hosts, high-risk TLDs, suspicious keywords, and obfuscated encodings before reaching a verdict of legitimate or phish.",
   # # 3 Few-Shot Contrastive Examples Prompt
   # "Here are four labeled examples to guide your reasoning:\n"
   # "legitimate: https://www.amazon.com/gp/help/customer/display.html\n"
   # "legitimate: https://accounts.google.com/ServiceLogin\n"
   # "phish:      http://secure-amazon-login.com.verify-pay.info/login.php\n"
   # "phish:      http://google-account.security-update.xyz/authenticate\n"
   # "Use the patterns you observe to judge the next URL.",
   # # 4 Hidden Chain-of-Thought Prompt
   # "Silently consider red flags such as brand spoofing, punycode, and unusual subdomains, then decide whether the URL is legitimate or phish.",
   # # 5 Scoring Rubric Prompt
   # "Assign one point each for an IP host, more than three subdomains, a high-risk keyword, or a path longer than 50 characters; two or more points indicates phishing.",
   # # 6 Heuristic-First Cascade Prompt
   # "If any structural heuristic—IP host, recent registration, or risky TLD—applies, treat the URL as phishing; otherwise, inspect phishing keywords before deciding.",
   # # 7 Explicit Negative Instruction Prompt
   # "Base your judgment only on the characters visible in the URL; ignore WHOIS, page content, and external reputation feeds.",
   # # 8 Token Breakdown Prompt
   # "List domain and path tokens, mark any that match risky words, and then determine whether the URL is legitimate or phish.",
   # # 9 Entropy & Length Hint Prompt
   # "Treat random-looking hosts or paths with high entropy and unusually long URLs as strong phishing indicators when making your decision.",
   # # 10 Dual-Model Emulation Prompt
   # "First estimate similarity to a phishing corpus using a TF-IDF-like approach, then adjust with heuristic penalties such as IP host or punycode to reach a final judgment." ,
   #
   # ## Reduce FP
   # # 11 Conservative Threshold Prompt
   # "Decide whether the following URL is legitimate or phish. Only label a URL as phish if it shows clear indicators such as brand impersonation, suspicious keywords, high-risk TLDs, or unusual subdomain depth; if evidence is weak or ambiguous, classify it as legitimate.",
   # # 12 Two-Red-Flags Prompt
   # "Decide whether the following URL is legitimate or phish. If you detect at least two phishing red flags (e.g., punycode, IP address host, credential keywords), mark it as phish; otherwise treat it as legitimate.",
   # # 13 Keyword-AND-Structure Prompt
   # "Decide whether the following URL is legitimate or phish. Flag as phish only when both a sensitive keyword (login, verify, update, reset) AND a structural anomaly (IP host, high-risk TLD, more than three subdomains) are present.",
   # # 14 Score-Based Prompt
   # "Decide whether the following URL is legitimate or phish. Give one point for each of these features: suspicious keyword, risky TLD, IP host, path longer than 60 characters. Require at least two points before labelling phish; otherwise classify as legitimate.",
   # # 15 Uncertainty-Bias-Legit Prompt
   # "Decide whether the following URL is legitimate or phish. When uncertain, err on the side of legitimate and classify phishing only when strong evidence exists."


)

# ─── BASE + SUFFIX PROMPTS ───────────────────────────────────────────
#
#UFFIXES = [
#   # 1
#   """
#   Now evaluate the raw URL text for excessive length—domains over 70 characters often signal auto-generated phishing. Count hyphens and underscores; more than three in the hostname suggests obfuscation. Identify long numeric runs (e.g., '1234567890') or sequences of random letters mixed with digits. Compute character-level Shannon entropy: values above 4.0 typically indicate randomness rather than meaningful words. Split the domain on delimiters and calculate the dictionary-word ratio—URLs with less than 30% real words and over 70% random tokens are suspicious. Compare the TLD against a rarity list: extensions used by fewer than 0.05% of legitimate sites (e.g., .club, .win) raise risk. Check for chained extensions (e.g., '.com.top') and private-use or unassigned TLD patterns. Flag any URL matching two or more of these heuristics before concluding.
#   """,
#   # 2
#   """
#   Next, measure the Levenshtein distance between the second-level domain and a set of top brands—distances of 1 or 2 often reflect typo-squatting. Scan for Unicode homoglyphs: characters like Cyrillic 'а' instead of Latin 'a' or Greek 'ο' replacing 'o'. Count uppercase vs. lowercase transitions; more than five random case flips indicate automated generation. Calculate subdomain depth—URLs with four or more segments (e.g., a.b.c.d.example.com) are riskier. Detect literal port numbers in the hostname (':8080', ':8443') which rarely appear in legitimate brand domains. Spot chained file or TLD sequences ('.net.exe', '.com.xyz') and flag them. Identify numeric-only labels (tokens longer than five digits). If three or more of these anomalies occur, classify as phishing.
#   """,
#   # 3
#   """
#   Analyze the token composition: split on dots, dashes, and underscores, then classify each token as dictionary word or random string. A ratio of over 60% random tokens is a strong phishing indicator. Look for repeated patterns (e.g., 'loginlogin', 'securesecure') which suggest automated scripts. Identify hyphen clusters—three or more hyphens consecutively signal obfuscation. Compute the punctuation-to-alphanumeric ratio: if symbols constitute more than 20% of characters, raise suspicion. Assess the entropy of the TLD string itself: unusual or lengthy TLD names with high entropy differ from standard four- or five-character TLDs. Verify that no token is purely numeric unless representing a valid country code or port. If at least five of these checks fail, mark the URL as phishing.
#   """,
#   # 4
#   """
#   Examine the position of dictionary words relative to the brand token—legitimate URLs place the brand at the root, not in deep paths. Detect digit-letter interleaving (e.g., 'payp4l2secure'), which often reflects evasion. Compare each label’s spelling to a brand whitelist using fuzzy matching; misspellings like 'paypaal' or 'amaz0n' within edit distance 1 count as impersonation. Infer TLD age by its length and pattern: newly popular extensions like '.icu' or '.xin' often correlate with short-lived domains. Identify tokens that combine brand with descriptive terms ('secure', 'update', 'verify') and ensure they appear logically (prefix vs. suffix). Flag URLs where brand-like tokens appear only after the first dot or within query parameters. Conclude phishing if three or more proximity or misspelling heuristics trigger.
#   """,
#   # 5
#   """
#   Finally, detect clusters of uncommon symbols (e.g., '~', '!', '@') adjacent to the brand token—these indicate obfuscation layers. Scan for camelCase patterns in the domain (e.g., 'PayPalSecureLogin') which rarely appear in official hostnames. Measure the length of numeric suffixes—sequences longer than four digits following the brand (e.g., 'paypal12345') are autogenerated. Identify domains containing two or more distinct brand names concatenated (e.g., 'paypalamazon-login') as a clear phishing signal. Count transitions between character classes (letter→digit, digit→symbol) and flag URLs exceeding ten transitions. Check for non-ASCII characters anywhere in the hostname, which phishing kits use to evade detection. If four or more of these symbol and pattern heuristics are present, treat the URL as phishing.
#   """
#
#TASK_PROMPTS = [BASE_PROMPT + " " + suffix.strip() for suffix in SUFFIXES]

SYS_PROMPTS = BASE_PROMPT

# ─── LOGGING ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ─── HELPERS ─────────────────────────────────────────────────────────

def normalise_gt(label: str) -> str:
    l = label.strip().lower()
    return "phish" if l in {"phish", "phishing", "bad", "malicious"} else "legitimate"

async def call_llama3_async(session: aiohttp.ClientSession, prompt: str,
                            max_tokens: int = 2,
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
    if "text" in data:
        return data["text"]
    if "choices" in data and isinstance(data["choices"], list):
        return data["choices"][0].get("text", "")
    return ""

async def classify_url_async(session: aiohttp.ClientSession, url_text: str, system_text: str
                            ) -> Tuple[str, str]:
    prompt = (
        "### SYSTEM ###\n" +
        system_text +
        "\nRespond with exactly ONE word 'phish' or 'legitimate', after ### ANSWER ###. no other tokens." + "\n\n" 
        "### URL ###\n\n" + url_text +
        "\n\n ### ANSWER ###\n\n"

    )
    raw_out = await call_llama3_async(session, prompt)
    parts = re.split(r"###\s*EXPLANATION\s*###", raw_out, flags=re.IGNORECASE)
    before = parts[0]
    m = re.search(r"###\s*(?:ANSWER|RESPONSE)\s*###\s*(\w+)", before, re.IGNORECASE)
    if m:
        lab = m.group(1).lower()
    else:
        m2 = re.search(r"\b(legitimate|phish)\b", before, re.IGNORECASE)
        lab = m2.group(1).lower() if m2 else "unknown"
    return (lab if lab in {"legitimate", "phish"} else "unknown", raw_out)

def plot_confusion_matrix(tp: int, fp: int, tn: int, fn: int, save_path: Path) -> None:
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0,1], labels=["Pred legit","Pred phish"])
    ax.set_yticks([0,1], labels=["GT legit","GT phish"])
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ─── MAIN ────────────────────────────────────────────────────────────

async def main():
    logging.info(f"Reading CSV from {CSV_PATH}")
    rows: List[Tuple[str,str]] = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.reader(f), 1):
            if not row or len(row) < 2:
                logging.warning(f"Skipping malformed row {idx}: {row}")
                continue
            rows.append((row[0].strip(), normalise_gt(row[1])))

    all_summaries = []
    async with aiohttp.ClientSession() as session:
        for idx, sys in enumerate(SYS_PROMPTS, start=1):
            tp = fp = tn = fn = skipped = 0
            per_url_results: List[Dict[str,Any]] = []

            out_json = OUTPUT_DIR / f"llm_results_summary_{idx}.json"
            out_png  = OUTPUT_DIR / f"llm_confusion_matrix_{idx}.png"

            print(f"\n=== Prompt #{idx} ===")
            for i, (url, gt) in enumerate(rows, 1):
                print(f"[{i}/{len(rows)}] {url}")
                trials = []
                for t in range(1, NUM_TRIES+1):
                    pred, raw = await classify_url_async(session, url, sys)
                    trials.append({"raw": raw, "extracted": pred})
                    short = raw.replace("\n", " ")[:100] + ("..." if len(raw)>100 else "")
                    print(f"  Trial {t}: extracted={pred} | raw=\"{short}\"")

                final = max((d["extracted"] for d in trials), key=lambda x: trials.count({"extracted":x}) if x!="unknown" else -1)
                print(f"   → Final: {final} (GT: {gt})")

                if final == "unknown":
                    skipped += 1
                else:
                    if gt=="phish" and final=="phish":  tp += 1
                    elif gt=="phish" and final=="legitimate": fn += 1
                    elif gt=="legitimate" and final=="legitimate": tn += 1
                    else: fp += 1

                per_url_results.append({
                    "url": url,
                    "ground_truth": gt,
                    "final_prediction": final,
                    "trials": trials
                })

            # save detailed per-run summary
            summary = {
                "prompt": sys,
                "counts": {"TP":tp,"FP":fp,"TN":tn,"FN":fn,"unknown":skipped},
                "precision": tp/(tp+fp)    if tp+fp else 0.0,
                "recall":    tp/(tp+fn)    if tp+fn else 0.0,
                "f1":        2*(tp/(tp+fp))*(tp/(tp+fn))/((tp/(tp+fp))+(tp/(tp+fn))) if tp and (tp+fp) and (tp+fn) else 0.0,
                "fpr":       fp/(fp+tn)    if fp+tn else 0.0,
                "num_samples": len(rows),
                "per_url_results": per_url_results
            }
            out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logging.info(f"Saved summary to {out_json}")

            if any([tp,fp,tn,fn]):
                plot_confusion_matrix(tp, fp, tn, fn, out_png)
                logging.info(f"Saved CM to {out_png}")
            else:
                logging.warning("All predictions unknown; skipped plotting.")

            # collect reduced summary for combined file
            all_summaries.append({
                "prompt":      sys,
                "counts":      summary["counts"],
                "precision":   summary["precision"],
                "recall":      summary["recall"],
                "f1":          summary["f1"],
                "fpr":         summary["fpr"],
                "num_samples": summary["num_samples"]
            })

    # write combined summary without per_url_results
    combo_path = OUTPUT_DIR / "all_llm_summary.json"
    combo = {"runs": all_summaries}
    combo_path.write_text(json.dumps(combo, indent=2), encoding="utf-8")
    logging.info(f"Wrote combined summary to {combo_path}")

if __name__ == "__main__":
    asyncio.run(main())
