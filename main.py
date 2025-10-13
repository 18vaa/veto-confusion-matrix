# src/confusion_matrix.py
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv


# ============================================
# CONFIG — edit these in one place
# ============================================
# Input/Output paths (use absolute paths if you prefer)
INPUT_PATH = Path(
    r"D:\Vetology\Research Student Assignments\Research Student Assignments\Input Data 2 - feline_thorax_scoring.xlsx"
)
OUTPUT_PATH = Path(r"D:\Vetology\Research Student Assignments\Research Student Assignments\confusion_output.xlsx")

# Process only first N rows (set to None for all rows)
N_ROWS: Optional[int] = 50  # e.g., 10 while testing

# Column names in your sheet
RAD_TXT_COL = "Findings (original radiologist report)"
AI_TXT_COL = "Findings (AI report)"
CASE_ID_COL = "CaseID"  # if missing, sequential IDs will be used

# Exclude these placeholders and narrative fields from label block
PLACEHOLDERS = {"True_Pos", "True_Neg", "False_Pos", "False_Neg"}

# Default scorer: "gpt" (real) | "gemini" | "mock"
MODEL = "gpt"

# Optional: clip AI text to reduce tokens
MAX_AI_CHARS = 1500
# ============================================


# ------------- env -------------
def init_env() -> None:
    """Load environment variables from .env (OPENAI_API_KEY / GEMINI_API_KEY)."""
    load_dotenv()


# ------------- IO + label detection -------------
def load_dataframe(path: Path, sheet: int | str = 0, n_rows: Optional[int] = N_ROWS) -> pd.DataFrame:
    """Read Excel, honoring N_ROWS for speed during testing."""
    if n_rows is not None:
        df = pd.read_excel(path, sheet_name=sheet, nrows=n_rows)
    else:
        df = pd.read_excel(path, sheet_name=sheet)
    return df


def detect_label_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect label columns as those to the right of AI findings,
    excluding placeholders and any narrative/AI meta columns.
    """
    cols = list(df.columns)
    if AI_TXT_COL not in cols:
        raise KeyError(f"AI text column not found: {AI_TXT_COL}")

    ai_idx = cols.index(AI_TXT_COL)

    NON_LABELS = {
        AI_TXT_COL,
        RAD_TXT_COL,
        "Conclusions (AI report)",
        "Recommendations (AI report)",
        "Conclusions (original radiologist report)",
        "Recommendations (original radiologist report)",
        *PLACEHOLDERS,
    }

    candidate_cols = cols[ai_idx + 1 :]
    label_cols = [
        c
        for c in candidate_cols
        if c not in NON_LABELS and "(AI report)" not in c and "(original radiologist report)" not in c
    ]

    if not label_cols:
        raise ValueError("No label columns detected to the right of AI findings after filtering.")
    return label_cols


# ------------- ground truth normalization (radiologist labels) -------------
def normalize_label(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    s = str(value).strip().lower()
    if s in {"abnormal", "present", "positive", "yes", "1", "true"}:
        return "Positive"
    if s in {"normal", "absent", "negative", "no", "0", "false"}:
        return "Negative"
    return "Unknown"


def build_ground_truth(df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    """Map raw strings (Normal/Abnormal/etc.) to Positive/Negative/Unknown."""
    return df[label_cols].applymap(normalize_label)


# ------------- AI scorers -------------
def _prompt_ai(ai_text: str, labels: List[str]) -> str:
    # Strict, fence-free JSON request
    return (
        "Return JSON ONLY with no markdown or code fences:\n"
        '{"per_condition": {"<label>":"Positive|Negative|Unknown"}}\n'
        "Rules: Positive=explicit presence/abnormality; Negative=explicit absence/normal; Unknown=not mentioned/unclear.\n"
        f"Labels: {json.dumps(labels, ensure_ascii=False)}\n"
        "AI Report:\n" + (ai_text or "")
    )


def extract_json(text: str) -> dict:
    """
    Strip ```json fences if present and return the largest JSON object in the text.
    Raises if none found.
    """
    if not text:
        raise ValueError("Empty model response.")

    # Remove leading/trailing code fences like ```json ... ```
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response head:\n{text[:400]}")

    payload = text[start : end + 1]
    return json.loads(payload)


def _normalize_pnu(v: str) -> str:
    v = (v or "").strip().lower()
    if v.startswith("pos"):
        return "Positive"
    if v.startswith("neg"):
        return "Negative"
    return "Unknown"


def call_gpt_ai(ai_text: str, labels: List[str]) -> Dict[str, str]:
    """Use OpenAI GPT to return {label: P/N/U}. Requires OPENAI_API_KEY in .env."""
    from openai import OpenAI  # lazy import so the package is optional

    api = os.getenv("OPENAI_API_KEY")
    if not api:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in your .env (and keep .env in .gitignore).")

    if not (ai_text or "").strip():
        return {lab: "Unknown" for lab in labels}

    if MAX_AI_CHARS:
        ai_text = ai_text[:MAX_AI_CHARS]

    client = OpenAI(api_key=api)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": _prompt_ai(ai_text, labels)}],
        temperature=0,
        max_tokens=600,  # room to list all labels
    )
    content = (resp.choices[0].message.content or "").strip()

    try:
        data = extract_json(content)
    except Exception as e:
        raise RuntimeError(f"Could not parse JSON from GPT:\n{content[:400]}") from e

    per = data.get("per_condition", {}) or {}
    return {lab: _normalize_pnu(per.get(lab, "Unknown")) for lab in labels}


def call_gemini_ai(ai_text: str, labels: List[str]) -> Dict[str, str]:
    """Use Google Gemini to return {label: P/N/U}. Requires GEMINI_API_KEY in .env."""
    import google.generativeai as genai  # lazy import

    api = os.getenv("GEMINI_API_KEY")
    if not api:
        raise RuntimeError("GEMINI_API_KEY not set. Put it in your .env.")

    if not (ai_text or "").strip():
        return {lab: "Unknown" for lab in labels}

    if MAX_AI_CHARS:
        ai_text = ai_text[:MAX_AI_CHARS]

    genai.configure(api_key=api)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(_prompt_ai(ai_text, labels))
    text = (resp.text or "").strip()

    try:
        data = extract_json(text)
    except Exception as e:
        raise RuntimeError(f"Could not parse JSON from Gemini:\n{text[:400]}") from e

    per = data.get("per_condition", {}) or {}
    return {lab: _normalize_pnu(per.get(lab, "Unknown")) for lab in labels}


def call_mock_ai(ai_text: str, labels: List[str]) -> Dict[str, str]:
    """Deterministic baseline: label is Positive if its name appears in the AI text; else Negative."""
    low = (ai_text or "").lower()
    return {lab: ("Positive" if lab.lower() in low else "Negative") for lab in labels}


def score_ai_frame(df_in: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    """Row-by-row AI predictions; returns a DataFrame with columns==labels and P/N/U values."""
    preds: List[Dict[str, str]] = []

    for _, row in df_in.iterrows():
        ai_text = str(row.get(AI_TXT_COL, "") or "")

        if MODEL == "gpt":
            per = call_gpt_ai(ai_text, labels)
        elif MODEL == "gemini":
            per = call_gemini_ai(ai_text, labels)
        else:
            per = call_mock_ai(ai_text, labels)

        preds.append(per)

    out = pd.DataFrame(preds)
    # ensure all expected label columns exist
    for lab in labels:
        if lab not in out.columns:
            out[lab] = "Unknown"
    return out[labels]


# ------------- confusion matrix + per-case -------------
def build_confusion_and_percase(
    gt_df: pd.DataFrame,
    ai_df: pd.DataFrame,
    labels: List[str],
    case_ids: pd.Series | List[Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate TP/FN/TN/FP per label and build per-case TP/TN/FP/FN lists."""
    counts: Dict[str, List[int]] = {lab: [0, 0, 0, 0] for lab in labels}  # tp, fn, tn, fp
    tp_s, tn_s, fp_s, fn_s = [], [], [], []

    for i in range(len(gt_df)):
        tps: List[str] = []
        tns: List[str] = []
        fps: List[str] = []
        fns: List[str] = []
        for lab in labels:
            g = gt_df.iloc[i][lab]
            p = ai_df.iloc[i][lab]
            if "Unknown" in (g, p):
                continue
            if g == "Positive" and p == "Positive":
                counts[lab][0] += 1
                tps.append(lab)
            elif g == "Positive" and p == "Negative":
                counts[lab][1] += 1
                fns.append(lab)
            elif g == "Negative" and p == "Negative":
                counts[lab][2] += 1
                tns.append(lab)
            elif g == "Negative" and p == "Positive":
                counts[lab][3] += 1
                fps.append(lab)
        tp_s.append(", ".join(tps))
        tn_s.append(", ".join(tns))
        fp_s.append(", ".join(fps))
        fn_s.append(", ".join(fns))

    # build matrix rows with NA when denominator is zero (undefined)
    rows = []
    for lab, (tp, fn, tn, fp) in counts.items():
        sens = tp / (tp + fn) if (tp + fn) > 0 else None
        spec = tn / (tn + fp) if (tn + fp) > 0 else None
        rows.append(
            {
                "condition": lab,
                "tp_Positive": tp,
                "fn_Positive": fn,
                "tn_Positive": tn,
                "fp_Positive": fp,
                "Sensitivity": f"{sens:.2%}" if sens is not None else "NA",
                "Specificity": f"{spec:.2%}" if spec is not None else "NA",
                "Check": tp + fn + tn + fp,
                "Positive Ground Truth": tp + fn,
                "Negative Ground Truth": tn + fp,
                "Ground Truth Check": tp + fn + tn + fp,
            }
        )

    matrix = pd.DataFrame(rows)[
        [
            "condition",
            "tp_Positive",
            "fn_Positive",
            "tn_Positive",
            "fp_Positive",
            "Sensitivity",
            "Specificity",
            "Check",
            "Positive Ground Truth",
            "Negative Ground Truth",
            "Ground Truth Check",
        ]
    ]

    case_ids_series = pd.Series(case_ids).reset_index(drop=True)
    per_case = pd.DataFrame(
        {
            "CaseID": case_ids_series,
            "True_Pos": tp_s,
            "True_Neg": tn_s,
            "False_Pos": fp_s,
            "False_Neg": fn_s,
        }
    )

    return matrix, per_case


# ------------- write output -------------
def write_output(matrix_df: pd.DataFrame, per_case_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        matrix_df.to_excel(w, index=False, sheet_name="ConfusionMatrix")
        per_case_df.to_excel(w, index=False, sheet_name="PerCase")


# ------------- main -------------
def main() -> None:
    init_env()

    # load
    df = load_dataframe(INPUT_PATH, sheet=0, n_rows=N_ROWS)  # note: param name fix below
    # If you copied older signature, keep the original call:
    # df = load_dataframe(INPUT_PATH, sheet=0, n_rows=N_ROWS)

    # Detect labels & build ground truth
    label_cols = detect_label_columns(df)
    gt = build_ground_truth(df, label_cols)

    # AI predictions
    ai_pred = score_ai_frame(df, label_cols)

    # confusion + per-case
    case_ids = df[CASE_ID_COL] if CASE_ID_COL in df.columns else pd.Series(range(len(df)))
    matrix_df, per_case_df = build_confusion_and_percase(gt, ai_pred, label_cols, case_ids)

    # write
    write_output(matrix_df, per_case_df, OUTPUT_PATH)
    print(f"✅ Wrote {OUTPUT_PATH}")
    print(f"Rows: {len(df)} | Labels: {len(label_cols)}")


if __name__ == "__main__":
    # Small backward-compat param name fix:
    # Ensure load_dataframe() accepts n_rows kwarg (as defined). If you pasted the older call,
    # either use n_rows= or rename the parameter in the function signature to nrows.
    main()
