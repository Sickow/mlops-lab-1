from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(raw_path: str) -> pd.DataFrame:
    raw_files = list(Path(raw_path).glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"Aucun CSV dans {raw_path}")
    return pd.read_csv(raw_files[0])

def split_xy(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def train_test_split_xy(X, y, test_size=0.2, random_state=42, stratify=True):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )
