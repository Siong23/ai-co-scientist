from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Columns that are identifiers, leakage-prone, or not useful as model inputs.
DEFAULT_DROP_COLUMNS = [
    "Seq", "Offset", "Label", "Attack Type", "Attack Tool"
]

@dataclass
class Preprocessor:
    feature_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    numeric_columns: Optional[List[str]] = None
    scaler: Optional[StandardScaler] = None
    category_maps: Optional[dict] = None
    label_encoder: Optional[LabelEncoder] = None

    def fit(self, df: pd.DataFrame, target="Label", drop_columns=None):
        drop = set(DEFAULT_DROP_COLUMNS if drop_columns is None else drop_columns)
        drop.discard(target)

        x = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore").copy()
        x = x.replace([np.inf, -np.inf], np.nan)

        self.categorical_columns = [
            c for c in x.columns
            if x[c].dtype == "object" or str(x[c].dtype).startswith("category")
        ]
        self.numeric_columns = [c for c in x.columns if c not in self.categorical_columns]

        self.category_maps = {}
        for c in self.categorical_columns:
            values = x[c].fillna("__MISSING__").astype(str).unique().tolist()
            self.category_maps[c] = {v: i for i, v in enumerate(values)}

        if self.numeric_columns:
            x_num = x[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
            x_num = x_num.replace([np.inf, -np.inf], np.nan)
            x_num = x_num.fillna(x_num.median()).fillna(0.0)
            self.scaler = StandardScaler()
            self.scaler.fit(x_num)

        self.feature_columns = list(x.columns)

        y = df[target].fillna("Benign").astype(str)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        return self

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        x = df.drop(columns=[c for c in DEFAULT_DROP_COLUMNS if c in df.columns], errors="ignore").copy()
        x = x.reindex(columns=self.feature_columns, fill_value=np.nan)
        x = x.replace([np.inf, -np.inf], np.nan)

        parts = []

        if self.numeric_columns:
            x_num = x[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
            x_num = x_num.replace([np.inf, -np.inf], np.nan)
            x_num = x_num.fillna(0.0)
            # Use training scaler. Unknown/missing numeric values become 0 after scaling
            # only when the fitted median is unavailable; normally the training median
            # should be used. This is kept simple for portability.
            parts.append(self.scaler.transform(x_num).astype(np.float32))

        if self.categorical_columns:
            cat = np.zeros((len(x), len(self.categorical_columns)), dtype=np.float32)
            for j, c in enumerate(self.categorical_columns):
                mapping = self.category_maps[c]
                cat[:, j] = [
                    mapping.get(str(v) if pd.notna(v) else "__MISSING__", -1)
                    for v in x[c]
                ]
            parts.append(cat)

        if not parts:
            raise ValueError("No usable feature columns were found.")
        return np.concatenate(parts, axis=1).astype(np.float32)

    def transform_labels(self, df: pd.DataFrame, target="Label") -> np.ndarray:
        y = df[target].fillna("Benign").astype(str)
        return self.label_encoder.transform(y)

    @property
    def n_features(self):
        return len(self.numeric_columns or []) + len(self.categorical_columns or [])

    @property
    def classes(self):
        return list(self.label_encoder.classes_)

def make_binary_target(df: pd.DataFrame, label_column="Label") -> pd.DataFrame:
    out = df.copy()
    out[label_column] = (
        out[label_column].astype(str).str.strip().str.lower()
        .eq("benign")
        .map({True: "Benign", False: "Attack"})
    )
    return out

def train_test_split_stratified(df, label_column="Label", test_size=0.30, random_state=42):
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column].astype(str)
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
