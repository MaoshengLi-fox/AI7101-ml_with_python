import random
import numpy as np

import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def set_seed(seed: int = 42):
  random.seed(seed)
  np.random.seed(seed)

def evaluate(gscv,X_test,y_test):
  best = gscv.best_estimator_
  y_pred = best.predict(X_test)
  rmse = mean_squared_error(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  return y_pred,best,rmse,mae,r2
def plot_img(y_test,y_pred):
  os.makedirs("figures", exist_ok=True)
  # Plots — keep visualization in notebook after you refactor logic into src/
  plt.figure()
  plt.hist(y_test - y_pred, bins=40)
  plt.title("Residuals (y - y_pred)")
  plt.xlabel("Residual")
  plt.ylabel("Count")
  plt.tight_layout()
  plt.savefig(os.path.join("figures", "residuals.png"))
  plt.show()

  plt.figure()
  plt.scatter(y_test, y_pred, s=6)
  plt.title("Predicted vs True")
  plt.xlabel("True")
  plt.ylabel("Pred")
  plt.tight_layout()
  plt.savefig(os.path.join("figures", "pred_vs_true.png"))
  plt.show()