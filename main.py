# Import library yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1. Membuat dataset sederhana
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Variabel independen (X)
Y = 3 * X + np.random.randn(100, 1) * 2  # Variabel dependen (Y) dengan sedikit noise

# Menampilkan beberapa sampel data
print("Sampel Data:")
print(pd.DataFrame(np.hstack((X, Y)), columns=["X", "Y"]).head())

# 2. Membagi dataset menjadi training dan testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 3. Membuat model Linear Regression
model = LinearRegression()
model.fit(X_train, Y_train) # Train Model

# 4. Melihat hasil model
print("\nHasil Training Model:")
print(f"Intercept (b0): {model.intercept_[0]:.2f}")
print(f"Koefisien (b1): {model.coef_[0][0]:.2f}")

# 5. Prediksi pada data uji
Y_pred = model.predict(X_test)

# 6. Evaluasi model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\nEvaluasi Model:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")

# 7. Visualisasi hasil regresi
plt.scatter(X_test, Y_test, color="blue", label="Actual Data")
plt.plot(X_test, Y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Hasil Linear Regression")
plt.show()

# Simpan ke file karena plt.show() tidak tampil di Codespaces
plt.savefig("regression_plot.png")
print("Plot disimpan sebagai 'regression_plot.png'")

# Menyimpan hasil train model
with open("models/linear_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model berhasil disimpan sebagai 'linear_model.pkl'")
