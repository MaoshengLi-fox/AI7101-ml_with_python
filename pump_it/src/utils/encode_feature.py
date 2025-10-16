import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class RareCategoryGrouperSimple(BaseEstimator, TransformerMixin):
    """
      Merge low-frequency data by column;
      save the top_k categories in fit and merge the rest into 'Other' during transform。
    """
    def __init__(self, top_k: int = 100):
        self.top_k = top_k
        self.keep_maps_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.keep_maps_ = {}
        for c in X.columns:
            counts = X[c].astype('object').value_counts()
            keep = set(counts.head(self.top_k).index)
            self.keep_maps_[c] = keep
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            keep = self.keep_maps_.get(c, set())
            X[c] = X[c].astype('object').where(X[c].isin(keep), other="Other")
        return X.values

class FrequencyEncoderSimple(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.freq_maps_ = {}
        for c in X.columns:
            freq = X[c].astype('object').value_counts(normalize=True)
            self.freq_maps_[c] = freq.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            fmap = self.freq_maps_.get(c, {})
            X[c] = X[c].astype('object').map(fmap).fillna(0.0)
        return X.values

def build_preprocessor(X, *, high_card_threshold: int = 50, rare_top_k: int = 100):
    """
    Split categorical columns based on cardinality:
      - High cardinality: RareCategoryGrouperSimple + FrequencyEncoderSimple
      - Low/medium cardinality: RareCategoryGrouperSimple + One-Hot
    return: (preprocessor, num_cols, cat_low_cols, cat_high_cols)
    """
    # numeric columns, excluding id
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if "id" in num_cols:
        num_cols.remove("id")

    # columns of object or category type, excluding date_recorded
    cat_all = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if "date_recorded" in cat_all:
        cat_all.remove("date_recorded")

    # Split high/low cardinality
    nunique_map = {c: X[c].nunique(dropna=True) for c in cat_all}
    cat_high = [c for c in cat_all if nunique_map[c] > high_card_threshold]
    cat_low = [c for c in cat_all if c not in cat_high]

    # numeric pipeline
    numeric = Pipeline([
        ("scaler", StandardScaler()),
    ])

    # One-Hot
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=20)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    # Low/medium cardinality: low-frequency merging -> One-Hot
    cat_low_pipeline = Pipeline([
        ("rare", RareCategoryGrouperSimple(top_k=rare_top_k)),
        ("onehot", onehot),
    ])

    # High cardinality: low frequency merging -> frequency encoding
    cat_high_pipeline = Pipeline([
        ("rare", RareCategoryGrouperSimple(top_k=rare_top_k)),
        ("freq", FrequencyEncoderSimple()),
    ])

    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat_low", cat_low_pipeline, cat_low),
        ("cat_high", cat_high_pipeline, cat_high),
    ])

    return pre, num_cols, cat_low, cat_high