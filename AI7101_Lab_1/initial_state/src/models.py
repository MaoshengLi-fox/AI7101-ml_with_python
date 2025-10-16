from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet

def create_simple_linear_model():
  param_grid = {
    "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "model__l1_ratio": [0.0, 0.5, 1.0],
  }
  model = ElasticNet(max_iter=1000)

  pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])

  return param_grid,pipe

def create_Polynomial_model():
  param_grid = {
      "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
      "model__l1_ratio": [0.0, 0.5, 1.0],
  }
  model = ElasticNet(max_iter=1000)

  pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("poly", PolynomialFeatures(degree=2, include_bias=False)), ("scaler", StandardScaler()), ("model", model)])

  return param_grid,pipe

def create_KNN_model():
  param_grid = {
      "model__n_neighbors": [2,5,10,20,50],
      "model__p": [1,2],
  }
  model = KNeighborsRegressor()

  pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])

  return param_grid,pipe

def create_model(model_name):
  if model_name=='simple_linear':
    return create_simple_linear_model()
  elif model_name =='polynomial':
    return create_Polynomial_model()
  elif model_name=='knn':
    return create_KNN_model()
