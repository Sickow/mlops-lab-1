from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def add_combined_feature(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "mean radius" in X.columns and "mean texture" in X.columns:
        X["Combined_radius_texture"] = X["mean radius"] * X["mean texture"]
    return X

def build_preprocessor(num_cols, cat_cols, use_feature_eng: bool = True):
    num_steps = []
    if use_feature_eng:
        num_steps.append(("feat_eng", FunctionTransformer(add_combined_feature)))
    num_steps.append(("scaler", StandardScaler()))
    pre_num = Pipeline(steps=num_steps)

    pre = ColumnTransformer(
        transformers=[
            ("num", pre_num, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return pre
