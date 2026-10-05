# import pandas as pd

# df = pd.read_csv(
#     r"C:\xampp\htdocs\ai-co-scientist-1\data\5g_nidd\5g_nidd.csv",
#     low_memory=False
# )

# print(df["Attack Type"].value_counts(dropna=False))


import pandas as pd
from pathlib import Path

# Change this to your 5G-NIDD CSV path
DATASET_PATH = Path(r"data\5g_nidd\5g_nidd.csv")

# Number of rows to inspect
SAMPLE_SIZE = 10

print("=" * 80)
print("5G-NIDD DATASET INSPECTION")
print("=" * 80)

# ---------------------------------------------------------
# 1. Check file
# ---------------------------------------------------------
if not DATASET_PATH.exists():
    print(f"\n❌ Dataset not found:")
    print(DATASET_PATH)
    raise SystemExit(1)

print(f"\nDataset: {DATASET_PATH}")
print(f"File size: {DATASET_PATH.stat().st_size / (1024**3):.2f} GB")

# ---------------------------------------------------------
# 2. Read only a small sample
# ---------------------------------------------------------
print("\nReading sample...")

df = pd.read_csv(
    DATASET_PATH,
    nrows=SAMPLE_SIZE,
    low_memory=False
)

# ---------------------------------------------------------
# 3. Dataset layout
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("COLUMN INFORMATION")
print("=" * 80)

print(f"\nNumber of columns: {len(df.columns)}")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")

# ---------------------------------------------------------
# 4. First rows
# ---------------------------------------------------------
print("\n" + "=" * 80)
print(f"FIRST {SAMPLE_SIZE} ROWS")
print("=" * 80)

print(df.to_string(index=False))

# ---------------------------------------------------------
# 5. Data types
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)

print(df.dtypes.to_string())

# ---------------------------------------------------------
# 6. Missing values in sample
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("MISSING VALUES IN SAMPLE")
print("=" * 80)

missing = df.isnull().sum()

for column, count in missing.items():
    if count > 0:
        print(f"{column}: {count}")

if missing.sum() == 0:
    print("No missing values found in the sample.")

# ---------------------------------------------------------
# 7. Identify possible label columns
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("POSSIBLE LABEL / TARGET COLUMNS")
print("=" * 80)

label_keywords = [
    "label",
    "attack",
    "class",
    "target",
    "category",
    "type"
]

possible_labels = [
    column
    for column in df.columns
    if any(keyword in column.lower() for keyword in label_keywords)
]

if possible_labels:
    for column in possible_labels:
        print(f"\nColumn: {column}")
        print(df[column].value_counts(dropna=False).to_string())
else:
    print("No obvious label column found.")

# ---------------------------------------------------------
# 8. Numeric columns
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("NUMERIC COLUMNS")
print("=" * 80)

numeric_columns = df.select_dtypes(include="number").columns.tolist()

print(f"Number of numeric columns: {len(numeric_columns)}")

for column in numeric_columns:
    print(f"- {column}")

# ---------------------------------------------------------
# 9. Categorical/object columns
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("NON-NUMERIC COLUMNS")
print("=" * 80)

categorical_columns = df.select_dtypes(
    exclude="number"
).columns.tolist()

for column in categorical_columns:
    print(f"- {column}")

# ---------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Columns          : {len(df.columns)}")
print(f"Sample rows      : {len(df)}")
print(f"Numeric columns  : {len(numeric_columns)}")
print(f"Other columns    : {len(categorical_columns)}")

print("\nInspection completed.")