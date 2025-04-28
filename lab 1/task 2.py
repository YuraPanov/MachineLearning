import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

print("Названия признаков:", feature_names)

bmi_index = feature_names.index('bmi')

X_bmi = X[:, bmi_index].reshape(-1, 1)

model_sklearn = LinearRegression()
model_sklearn.fit(X_bmi, y)

coef_sklearn = model_sklearn.coef_[0]
intercept_sklearn = model_sklearn.intercept_

print(f"Коэффициенты Scikit-Learn: коэффициент = {coef_sklearn:.4f}, свободный член = {intercept_sklearn:.4f}")

X_flat = X_bmi.flatten()
b_manual = np.cov(X_flat, y, bias=True)[0, 1] / np.var(X_flat)
a_manual = np.mean(y) - b_manual * np.mean(X_flat)

print(f"Коэффициенты вручную: коэффициент = {b_manual:.4f}, свободный член = {a_manual:.4f}")

plt.scatter(X_bmi, y, color='blue', label='Данные')
plt.plot(X_bmi, model_sklearn.predict(X_bmi), color='red', label='Scikit-Learn регрессия')
plt.plot(X_bmi, a_manual + b_manual * X_bmi, color='green', linestyle='--', label='Ручная регрессия')
plt.xlabel('BMI')
plt.ylabel('Target')
plt.title('Линейная регрессия по признаку BMI')
plt.legend()
plt.grid(True)
plt.show()

predictions_sklearn = model_sklearn.predict(X_bmi)
predictions_manual = a_manual + b_manual * X_flat

results_df = pd.DataFrame({
    'BMI': X_flat,
    'Истинное значение': y,
    'Предсказание (Sklearn)': predictions_sklearn,
    'Предсказание (Ручное)': predictions_manual
})

print("\nПервые 10 результатов предсказаний:")
print(results_df.head(10))

