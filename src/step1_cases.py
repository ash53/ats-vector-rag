import pandas as pd
import json

# -------------------------------
# Helper function to normalize text fields
# -------------------------------
def normalize_text(text):
    """
    Cleans and normalizes raw text fields to ensure consistency
    across resumes, job descriptions, and decision reasons.

    - Converts NaN values to empty strings
    - Strips leading/trailing whitespace
    - Converts text to lowercase
    - Replaces newlines and tabs with spaces
    """
    if pd.isna(text):
        return ""
    return (
        text.strip()
            .lower()
            .replace("\n", " ")
            .replace("\t", " ")
    )

# -------------------------------
# Main function to build canonical ATS cases
# -------------------------------
def build_cases(input_csv, output_jsonl):
    """
    Reads the raw recruitment CSV dataset and converts each row
    into a normalized, atomic CV-JD-Decision case.

    Each case represents one historical hiring evaluation and
    serves as the basic unit for downstream vector RAG and GraphRAG.
    """

    # Load the raw dataset
    df = pd.read_csv(input_csv)

    # Create a stable, human-readable case ID for each row
    # Example: case_00001, case_00002, ...
    df["case_id"] = df.index.map(lambda i: f"case_{i:05d}")

    cases = []

    # Iterate over each recruitment record
    for _, row in df.iterrows():
        # Construct a single canonical case object
        case = {
            # Unique identifier for this recruitment case
            "case_id": row["case_id"],

            # Job role (lowercased for normalization)
            "role": str(row.get("Role", "")).lower(),

            # Cleaned resume text (CV)
            "cv_text": normalize_text(row.get("Resume", "")),

            # Cleaned job description text (JD)
            "jd_text": normalize_text(row.get("Job_Description", "")),

            # Hiring decision (e.g., select / reject)
            "decision": str(row.get("decision", "")).lower(),

            # Free-text explanation for the decision
            "decision_reason": normalize_text(row.get("Reason_for_decision", "")),

            # Metadata for traceability and future extensions
            "metadata": {
                # Dataset source identifier
                "source": "kaggle_ai_recruitment",

                # Whether an interview transcript exists for this case
                "has_transcript": not pd.isna(row.get("Transcript"))
            }
        }

        # Add the case to the list
        cases.append(case)

    # Write all cases to a JSONL file (one JSON object per line)
    # This format is efficient, streamable, and RAG-friendly
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    # Build canonical recruitment cases from the raw CSV
    build_cases(
        "data/raw/dataset.csv",
        "data/processed/cases_stage1.jsonl"
    )
