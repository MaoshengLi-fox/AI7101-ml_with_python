"""Training, cross-validation, and evaluation helpers.

Example contents:
- MODELS dictionary defining model pipelines and grids for hyperparameter tuning.
- `train_model(...)` to run CV and return the fitted search object.
- `evaluate(...)` to compute metrics on a holdout set.
"""

from sklearn.model_selection import KFold, GridSearchCV

def train_model(param_grid,pipe,X_train,y_train):
  cv = KFold(n_splits=5, shuffle=True)
  gscv = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
  )
  gscv.fit(X_train, y_train)
  return gscv
