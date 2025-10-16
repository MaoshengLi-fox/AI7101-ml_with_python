"""Dataset loading and splitting utilities.

Example contents:
- `load_dataset(test_size: float, random_state: int)` that returns
  `(X_train, X_test, y_train, y_test)` for the chosen regression dataset.
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
def load_dataset(test_size: float, random_state: int):
  ds = fetch_california_housing(as_frame=True)
  df = ds.frame.copy()
  X = df.drop(columns=["MedHouseVal"])
  y = df["MedHouseVal"]
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size,random_state=random_state
  )
  return X_train, X_test, y_train, y_test