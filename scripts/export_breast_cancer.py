from sklearn.datasets import load_breast_cancer
import pandas as pd
from pathlib import Path

data = load_breast_cancer(as_frame=True)
df = pd.concat([data["data"], data["target"]], axis=1)
Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/breast_cancer.csv", index=False)
print("Saved -> data/raw/breast_cancer.csv")
