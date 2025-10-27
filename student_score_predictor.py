# Student Score Predictor (Regression)
# Requirements: pandas, numpy, scikit-learn, matplotlib
# pip install pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1) Sample dataset (Hours studied -> Score)
data = {
    "Hours": [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7,
              7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 4.0, 6.9, 7.8,
              6.1, 0.5, 9.5, 3.8, 4.8],
    "Score": [21, 47, 27, 75, 30, 20, 88, 60, 81, 25,
              85, 62, 41, 42, 17, 95, 30, 45, 76, 86,
              67, 12, 99, 38, 50]
}
df = pd.DataFrame(data)
print("Dataset (first 5 rows):")
print(df.head())

# 2) Split into features/target and train/test
X = df[["Hours"]].values
y = df["Score"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 4) Predict & evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel coefficient (slope): {model.coef_[0]:.4f}")
print(f"Model intercept: {model.intercept_:.4f}")
print(f"MSE: {mse:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

# 5) Show actual vs predicted
results = pd.DataFrame({
    "Hours": X_test.flatten(),
    "Actual Score": y_test,
    "Predicted Score": np.round(y_pred, 2),
    "Error": np.round(y_test - y_pred, 2)
}).sort_values(by="Hours", ascending=False).reset_index(drop=True)
print("\nTest set - Actual vs Predicted:")
print(results)

# 6) Plot scatter and regression line
plt.figure(figsize=(9,6))
plt.scatter(X, y)
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line)
plt.title("Student Scores vs Hours Studied")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.grid(True)
plt.show()

# 7) Example predictions for new inputs
new_hours = np.array([[4.5], [7.0], [9.0]])
preds = model.predict(new_hours)
for h, p in zip(new_hours.flatten(), preds):
    print(f"Hours={h} -> Predicted score = {p:.2f}")
