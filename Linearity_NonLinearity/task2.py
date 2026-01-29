import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from data import generate_xor_data
from visual import plot_2d_data

#Generate Non linear data
X,y=generate_xor_data(n=200)

plot_2d_data(X,y,title="XOR Data (Original Space)")

#Linear model without Feature transformation
linear_model=LogisticRegression()
linear_model.fit(X,y)
y_pred_linear=linear_model.predict(X)
linear_accuracy=accuracy_score(y,y_pred_linear)
print(f"Linear Model Accuracy on XOR data: {linear_accuracy:.2f}")
plot_2d_data(X,y_pred_linear,title="XOR Data - Linear model Predictions")

#Feature tranformation to polynomial features
poly= PolynomialFeatures(degree=2,include_bias=False)
X_poly=poly.fit_transform(X)
poly_model=LogisticRegression()
poly_model.fit(X_poly,y)
y_pred_poly=poly_model.predict(X_poly)
poly_accuracy=accuracy_score(y,y_pred_poly)
print(f"Polynomial Feature Model Accuracy on XOR data: {poly_accuracy:.2f}")
plot_2d_data(X,y_pred_poly,title="Poly model pred")
