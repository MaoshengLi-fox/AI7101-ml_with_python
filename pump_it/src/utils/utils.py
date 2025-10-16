import pandas as pd
import numpy as np
# Function to process date columns
def process_date_column(df, col_name):

    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')

    df[f"{col_name}_year"] = df[col_name].dt.year
    df[f"{col_name}_month"] = df[col_name].dt.month
    df[f"{col_name}_day"] = df[col_name].dt.day

    df.drop(columns=[col_name], inplace=True)

    return df

# Function to one-hot encode categorical columns
def one_hot_encode(df, categorical_cols, drop_first=False):
    df_encoded = pd.get_dummies(df,
                                columns=categorical_cols,
                                drop_first=drop_first,
                                dtype=np.uint8)
    return df_encoded

# Function to convert boolean columns to integers
def convert_bool_to_int(df, bool_cols=None):

    if bool_cols is None:
        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    for col in bool_cols:
        df[col] = df[col].astype(int)

    return df

# function to summarize unique values in each column that needs to be encoded
def unique_value_summary(df):
    cols = df.select_dtypes(include=["object", "category"]).columns

    summary = pd.DataFrame({
        "column": cols,
        "dtype": [df[c].dtype for c in cols],
        "nunique": [df[c].nunique(dropna=True) for c in cols],
    }).sort_values(by="nunique", ascending=False).reset_index(drop=True)

    return summary

def count_equal_rows(df, cols):

    if len(cols) < 2:
        raise ValueError("input cols must contain at least two columns")

    mask = df[cols].nunique(axis=1) == 1
    equal_count = mask.sum()

    return equal_count, mask

# define a set of tokens that represent missing values
MISSING_TOKENS = {"", " ", "na", "n/a", "none", "null", "nan", "unknown", "unk", "?", "-", "--"}

# function to normalize and replace missing value tokens with pd.NA
def normalize_missing_strings(df):
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip().str.lower()
            df[c] = df[c].replace(MISSING_TOKENS, pd.NA)
    return df

# Function to handle missing values
def handle_missing_values(df, label_col="status_group"):
    df = df.copy()

    # if construction_year is 0, treat it as missing
    if "construction_year" in df.columns:
        df["construction_year_missing"] = (df["construction_year"] == 0).astype(int)
        df["construction_year"] = df["construction_year"].replace(0, np.nan)

    # if categorical columns have missing values, fill with "Unknown"
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if label_col in cat_cols:
        cat_cols.remove(label_col)

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # if numeric columns have missing values, fill with median
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    return df
