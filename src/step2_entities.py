#Entity Extraction (Skills + Degrees + Certifications)
import json
import re
from pathlib import Path
from tqdm import tqdm

# Paths
INPUT_PATH = Path("data/processed/cases_stage1.jsonl")
OUTPUT_PATH = Path("data/processed/cases_stage2.jsonl")
SKILLS_PATH = Path("data/vocab/skills.txt")
CERTS_PATH = Path("data/vocab/certifications.txt")

# -------------------------
# Load vocabularies
# -------------------------
def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

SKILLS = load_vocab(SKILLS_PATH)
CERTIFICATIONS = load_vocab(CERTS_PATH)

# -------------------------
# Phrase extraction
# -------------------------
def extract_phrases(text, vocab):
    found = set()
    for term in vocab:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text):
            found.add(term)
    return sorted(found)

# -------------------------
# Degree extraction
# -------------------------
DEGREE_PATTERNS = {
    "bachelor": r"\bbachelor(?:'s)?\b",
    "master": r"\bmaster(?:'s)?\b",
    "phd": r"\bphd\b",
    "bsc": r"\bbsc\b",
    "msc": r"\bmsc\b",
    "computer science": r"\bcomputer science\b",
    "engineering": r"\bengineering\b",
    "information technology": r"\binformation technology\b"
}

def extract_degrees(text):
    found = set()
    for label, pattern in DEGREE_PATTERNS.items():
        if re.search(pattern, text):
            found.add(label)
    return sorted(found)

# -------------------------
# Main processing
# -------------------------
def main():
    assert INPUT_PATH.exists(), "Run step1_cases.py first!"

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Extracting skills, degrees, certifications"):
            case = json.loads(line)

            cv_text = case.get("cv_text", "")
            jd_text = case.get("jd_text", "")

            entities = {
                "skills_cv": extract_phrases(cv_text, SKILLS),
                "skills_jd": extract_phrases(jd_text, SKILLS),
                "degrees": extract_degrees(cv_text),
                "certifications": extract_phrases(cv_text, CERTIFICATIONS)
            }

            case["entities"] = entities
            fout.write(json.dumps(case) + "\n")

    print(f"Step 2 completed. Output written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
