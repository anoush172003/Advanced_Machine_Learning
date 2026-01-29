import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from data import generate_xor_data
from visual import plot_2d_data
from sklearn.svm import SVC
X,y=generate_xor_data(n=200)

plot_2d_data(X,y,title="XOR Data (Original Space)")

linear_svm=SVC(kernel='linear')
linear_svm.fit(X,y)
y_pred_linear=linear_svm.predict(X)
plot_2d_data(X,y_pred_linear,title="Linear SVM Predictions (Fails)")

poly_svm=SVC(kernel='poly',degree=2)
poly_svm.fit(X,y)
y_pred_poly=poly_svm.predict(X)
plot_2d_data(X,y_pred_poly,title="Poly SVM Predictions")

rbf_svm=SVC(kernel='rbf',gamma='scale')
rbf_svm.fit(X,y)
y_pred_rbf=rbf_svm.predict(X)
plot_2d_data(X,y_pred_rbf,title="RBF SVM Predictions")

print("Linear SVM Accuracy:",accuracy_score(y,y_pred_linear))
print("Polynomial Kernel SVM Accuracy:",accuracy_score(y,y_pred_poly))
print("RBF Kernel SVM Accuracy:",accuracy_score(y,y_pred_rbf))
