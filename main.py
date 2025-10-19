# main.py
"""
Main script to generate the confusion matrix for Feline Thorax scoring.
Uses helper functions from utils.py for clarity and reusability.
"""

from utils import read_input_excel, count_conditions, counts_to_dataframe, write_df_with_formulas

# -------------------------
# FILE PATHS
# -------------------------
input_file = (
    r"D:\Vetology\Research Student Assignments\Research Student Assignments\Input Data 2 - feline_thorax_scoring.xlsx"
)
output_file = r"D:\Vetology\Research Student Assignments\Research Student Assignments\confusion_output.xlsx"

# -------------------------
# CONDITIONS
# -------------------------
conditions = [
    "pulmonary_nodules",
    "esophagitis",
    "pneumonia",
    "bronchitis",
    "interstitial",
    "diseased_lungs",
    "hypo_plastic_trachea",
    "cardiomegaly",
    "pleural_effusion",
    "perihilar_infiltrate",
    "rtm",
    "focal_caudodorsal_lung",
    "right_sided_cardiomegaly",
    "focal_perihilar",
    "left_sided_cardiomegaly",
    "bronchiectasis",
    "pulmonary_vessel_enlargement",
    "thoracic_lymphadenopathy",
    "pulmonary_hypoinflation",
    "pericardial_effusion",
    "Fe_Alveolar",
]

# -------------------------
# COLUMN NAMES (matching Excel headers)
# -------------------------
col_names = {
    "tp": "True Positive",
    "fn": "False Negative",
    "tn": "True Negative",
    "fp": "False Positive",
}

# -------------------------
# STEP 1: Read Input Excel
# -------------------------
print("📘 Reading input file...")
df = read_input_excel(input_file)
print(f"✅ Loaded file with {len(df)} rows and {len(df.columns)} columns.")

# -------------------------
# STEP 2: Count Condition Outcomes
# -------------------------
print("🔍 Counting condition outcomes...")
counts = count_conditions(df, conditions, col_names=col_names)
print("✅ Condition counts computed successfully.")

# -------------------------
# STEP 3: Convert Counts to DataFrame
# -------------------------
print("🧮 Building output DataFrame...")
output_df = counts_to_dataframe(counts)
print("✅ Output DataFrame created.")

# -------------------------
# STEP 4: Write to Excel (with formulas)
# -------------------------
print("💾 Writing results to Excel...")
write_df_with_formulas(output_df, output_file)
print(f"✅ Output file created successfully at: {output_file}")

# -------------------------
# OPTIONAL: Preview Output
# -------------------------
print("\nPreview of output:")
print(output_df.head())
