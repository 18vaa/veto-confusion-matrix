# llm_utils.py
"""
Minimal LLM utilities to:
- build prompt (using provided template)
- call the LLM (one prompt per case)
- parse LLM markdown table -> pandas.DataFrame
- batch score a DataFrame of cases and return combined scored_df
- aggregate confusion counts per condition to the same dict format as utils.counts_to_dataframe expects
"""

from typing import List, Tuple, Dict, Optional
import os
import time
import logging
from io import StringIO

import pandas as pd
import re

# OpenAI client (adjust if you use another provider)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

logger = logging.getLogger("llm_utils")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# -------------------------
# PROMPT TEMPLATE (use exactly / similar to what you supplied)
# -------------------------
PROMPT_TEMPLATE = r"""
CONDITIONS:
pulmonary_nodules, esophagitis, pneumonia, bronchitis, interstitial, diseased_lungs,
hypo_plastic_trachea, cardiomegaly, pleural_effusion, perihilar_infiltrate, rtm,
focal_caudodorsal_lung, right_sided_cardiomegaly, focal_perihilar, left_sided_cardiomegaly,
bronchiectasis, pulmonary_vessel_enlargement, thoracic_lymphadenopathy, pulmonary_hypoinflation,
pericardial_effusion, Fe_Alveolar

CASE_ID: {case_id}

GROUND TRUTH: (per-condition values from spreadsheet)
{ground_truth_table}

AI_FINDINGS:
\"\"\"
{ai_findings}
\"\"\"

SCORING RULES (apply exactly):
- Ground_Truth mapping: 'Abnormal' -> 1, 'Normal' -> 0.
- Prediction mapping (from AI_Findings): 1 if the AI text **asserts** the condition (explicit term or common synonym) is present; 0 if it **does not assert** presence.
- **Negation handling:** If AI text states absence (e.g., "no evidence of pneumonia", "without pleural effusion", "negative for pneumothorax"), treat as Prediction = 0.
- **Hedging / possible language:** If AI text uses qualifiers like "possible", "cannot exclude", "suspicious for", or "consistent with", treat as Prediction = 1 (AI is reporting the finding).
- **Synonyms & short forms:** Recognize obvious synonyms (e.g., "cardiac enlargement" = cardiomegaly). Do not invent distant synonyms. If you are unsure whether a phrase matches a condition, prefer Prediction = 0.
- **Distinct tokens:** If a condition name is a substring of another (e.g., 'cardiomegaly' and 'right_sided_cardiomegaly'), treat them as distinct; match the specific token or phrase only.
- **Do not infer** clinical meaning not explicit in the AI_Findings. Only use the AI_Findings text to determine Prediction.
- **Evidence:** For any Prediction = 1 or for any FP/FN decision, include a 1–10 word excerpt from AI_Findings that justifies the decision. If none, leave Evidence empty.

DETERMINATION RULE (per condition):
- If Ground_Truth == 1 and Prediction == 1 -> Score = TP
- If Ground_Truth == 1 and Prediction == 0 -> Score = FN
- If Ground_Truth == 0 and Prediction == 1 -> Score = FP
- If Ground_Truth == 0 and Prediction == 0 -> Score = TN

OUTPUT FORMAT (markdown table only):
| Case_ID | Condition | Ground_Truth | Prediction | Score | Evidence |
|---------|-----------|--------------|------------|-------|----------|
| {case_id} | pulmonary_nodules | 0 | 0 | TN | "" |
| {case_id} | pneumonia | 1 | 1 | TP | "patchy alveolar consolidation" |
... (include one row for every condition listed)

Important: Output **only** the table — no explanations, no extra rows, no counts. Use exactly the column names shown.
"""


# -------------------------
# LLM CALL (simple wrapper)
# -------------------------
def call_llm(prompt: str, model: str = "gpt-5", temperature: float = 0.0, max_tokens: int = 1024, retries: int = 2, backoff: float = 1.0) -> str:
    """
    Minimal LLM call wrapper using OpenAI OpenAI() client.
    Adjust this function if you use a different SDK.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI client not available. Install openai package or modify call_llm for your provider.")

    client = OpenAI()
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a precise radiology scorer that outputs only a markdown table."},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # extract text defensively
            try:
                text = resp.choices[0].message.content
            except Exception:
                text = getattr(resp.choices[0], "text", str(resp))
            return text
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt}): {e}")
            if attempt > retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))


# -------------------------
# Parse markdown table -> DataFrame
# -------------------------
def _extract_table(text: str) -> str:
    """Return contiguous pipes table lines only."""
    lines = text.splitlines()
    table_lines = []
    started = False
    for line in lines:
        if line.strip().startswith("|"):
            table_lines.append(line)
            started = True
        else:
            if started:
                break
    return "\n".join(table_lines).strip()


def parse_case_table(raw_text: str) -> pd.DataFrame:
    """
    Parse an LLM response (markdown table) into DataFrame with columns:
    Case_ID, Condition, Ground_Truth, Prediction, Score, Evidence

    Raises ValueError if parsing fails.
    """
    table_text = _extract_table(raw_text)
    if not table_text:
        raise ValueError("No markdown table found in LLM response.")

    # read with pandas (pipe-separated)
    df = pd.read_csv(StringIO(table_text), sep="|").dropna(axis=1, how="all")
    # clean columns and drop empty header columns created by leading/trailing pipes
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, df.columns != ""]

    # strip whitespace from values
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # ensure required columns exist
    required = ["Case_ID", "Condition", "Ground_Truth", "Prediction", "Score", "Evidence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parsed table: {missing}")

    # coerce numeric flags
    df["Ground_Truth"] = df["Ground_Truth"].apply(lambda x: int(str(x).strip()) if str(x).strip().isdigit() else 0)
    df["Prediction"] = df["Prediction"].apply(lambda x: int(str(x).strip()) if str(x).strip().isdigit() else 0)

    return df[required]


# -------------------------
# Batch scoring
# -------------------------
def _build_ground_truth_table_for_row(row: pd.Series, conditions: List[str], gt_col_map: Dict[str, str]) -> str:
    """Produce newline 'cond: Normal/Abnormal' lines from row using gt_col_map mapping."""
    lines = []
    for cond in conditions:
        col = gt_col_map.get(cond, cond)
        val = row.get(col, "")
        if pd.isna(val) or str(val).strip() == "":
            val = "Normal"
        lines.append(f"{cond}: {str(val).strip()}")
    return "\n".join(lines)


def score_dataframe_with_llm(
    input_df: pd.DataFrame,
    case_id_col: str,
    ai_findings_col: str,
    gt_col_map: Dict[str, str],
    conditions: List[str],
    model: str = "gpt-5",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    retries: int = 2,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    For each row in input_df, build prompt, call LLM, parse table, append to results.
    Returns combined scored_df and a list of issues (case_id -> error).
    """
    results = []
    issues = []

    for idx, row in input_df.iterrows():
        case_id = str(row.get(case_id_col, idx))
        ai_findings = str(row.get(ai_findings_col, ""))

        ground_truth_table = _build_ground_truth_table_for_row(row, conditions, gt_col_map)
        prompt = PROMPT_TEMPLATE.format(case_id=case_id, ground_truth_table=ground_truth_table, ai_findings=ai_findings)

        try:
            raw = call_llm(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, retries=retries)
            df_case = parse_case_table(raw)
            # ensure Case_ID matches our case_id (or fix if LLM returned)
            df_case["Case_ID"] = df_case["Case_ID"].astype(str).fillna(case_id)
            results.append(df_case)
        except Exception as e:
            logger.error(f"Failed to score case {case_id}: {e}")
            issues.append({"case_id": case_id, "error": str(e)})
            # do not raise; continue to next row

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["Case_ID", "Condition", "Ground_Truth", "Prediction", "Score", "Evidence"])
    return combined, issues


# -------------------------
# Aggregate confusion counts
# -------------------------
def aggregate_confusion(scored_df: pd.DataFrame, conditions: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Given scored_df with Score in {TP,FP,FN,TN}, return dict:
    { condition: {"tp": int, "fn": int, "tn": int, "fp": int}, ... }
    """
    # initialize counts
    counts = {cond: {"tp": 0, "fn": 0, "tn": 0, "fp": 0} for cond in conditions}
    if scored_df.empty:
        return counts

    # group and count
    grouped = scored_df.groupby(["Condition", "Score"]).size().reset_index(name="count")
    for _, row in grouped.iterrows():
        cond = row["Condition"]
        score = row["Score"]
        cnt = int(row["count"])
        if cond not in counts:
            # if LLM returned unexpected condition, skip it
            continue
        if score == "TP":
            counts[cond]["tp"] += cnt
        elif score == "FN":
            counts[cond]["fn"] += cnt
        elif score == "TN":
            counts[cond]["tn"] += cnt
        elif score == "FP":
            counts[cond]["fp"] += cnt
    return counts
