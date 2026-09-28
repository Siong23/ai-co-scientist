import pandas as pd

df = pd.read_csv(
    r"C:\xampp\htdocs\ai-co-scientist-1\data\5g_nidd\5g_nidd.csv",
    low_memory=False
)

print(df["Attack Type"].value_counts(dropna=False))