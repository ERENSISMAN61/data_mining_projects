# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins

# load the Palmer Penguins dataset
penguins = load_penguins()

# select relevant numerical features
penguins = penguins[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]]

# handle missing values for numerical columns by filling with the most frequent value
imputer = SimpleImputer(strategy="most_frequent") 
numerical_columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
penguins[numerical_columns] = imputer.fit_transform(penguins[numerical_columns])

# handle missing values for categorical 'sex' column by filling with the mode (most frequent)
penguins['sex'] = penguins['sex'].fillna(penguins['sex'].mode()[0])
penguins_imputed = pd.get_dummies(penguins, columns=["sex"], drop_first=True)

# separate features (X) and target (y)
X = penguins_imputed.drop(columns=["body_mass_g"])
y = penguins_imputed["body_mass_g"]

# split data into training and testing sets for 1st test (test_size=0.5)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.5, random_state=42)

# scale the features to standardize the dataset for 1st test
scaler = StandardScaler()
X_train_scaled_1 = scaler.fit_transform(X_train_1)
X_test_scaled_1 = scaler.transform(X_test_1)

# split data into training and testing sets for 2nd test (test_size=0.2)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the features to standardize the dataset for 2nd test
X_train_scaled_2 = scaler.fit_transform(X_train_2)
X_test_scaled_2 = scaler.transform(X_test_2)

# 1st Test:
ridge_reg = Ridge(alpha=10)  
lasso_reg = Lasso(alpha=1)   
rf_reg = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42) 
linear_reg = LinearRegression() 

# fit models for the first test
linear_reg.fit(X_train_scaled_1, y_train_1)
ridge_reg.fit(X_train_scaled_1, y_train_1)
lasso_reg.fit(X_train_scaled_1, y_train_1)
rf_reg.fit(X_train_scaled_1, y_train_1)

# make predictions for the first test
y_pred_lr_1 = linear_reg.predict(X_test_scaled_1)
y_pred_ridge_1 = ridge_reg.predict(X_test_scaled_1)
y_pred_lasso_1 = lasso_reg.predict(X_test_scaled_1)
y_pred_rf_1 = rf_reg.predict(X_test_scaled_1)

# evaluate models for the first test
def evaluate_model(y_test, y_pred, model_name, test_number):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - Test {test_number} - MSE: {mse:.2f}, R^2: {r2:.2f}")

# evaluate models for the first test
evaluate_model(y_test_1, y_pred_lr_1, "Linear Regression", 1)
evaluate_model(y_test_1, y_pred_ridge_1, "Ridge Regression", 1)
evaluate_model(y_test_1, y_pred_lasso_1, "Lasso Regression", 1)
evaluate_model(y_test_1, y_pred_rf_1, "Random Forest Regression", 1)

# 2nd Test: Using hyperparameter optimization (improve models)
ridge_reg_2 = Ridge(alpha=0.1) 
lasso_reg_2 = Lasso(alpha=0.01)
rf_reg_2 = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=10, random_state=42)  

# Linear regression 
linear_reg_2 = LinearRegression() 

# fit models for the second test
ridge_reg_2.fit(X_train_scaled_2, y_train_2)
lasso_reg_2.fit(X_train_scaled_2, y_train_2)
rf_reg_2.fit(X_train_scaled_2, y_train_2)
linear_reg_2.fit(X_train_scaled_2, y_train_2)

# make predictions for the second test
y_pred_lr_2 = linear_reg_2.predict(X_test_scaled_2)
y_pred_ridge_2 = ridge_reg_2.predict(X_test_scaled_2)
y_pred_lasso_2 = lasso_reg_2.predict(X_test_scaled_2)
y_pred_rf_2 = rf_reg_2.predict(X_test_scaled_2)

# evaluate models for the second test
evaluate_model(y_test_2, y_pred_lr_2, "Linear Regression", 2)
evaluate_model(y_test_2, y_pred_ridge_2, "Ridge Regression", 2)
evaluate_model(y_test_2, y_pred_lasso_2, "Lasso Regression", 2)
evaluate_model(y_test_2, y_pred_rf_2, "Random Forest Regression", 2)

# plot the actual vs predicted values for each model (first test)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(y_test_1, y_pred_lr_1, color='blue')
plt.title("Linear Regression - Test 1")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 2)
plt.scatter(y_test_1, y_pred_ridge_1, color='green')
plt.title("Ridge Regression - Test 1")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 3)
plt.scatter(y_test_1, y_pred_lasso_1, color='red')
plt.title("Lasso Regression - Test 1")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 4)
plt.scatter(y_test_1, y_pred_rf_1, color='purple')
plt.title("Random Forest - Test 1")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.tight_layout()
plt.show()

# plot the actual vs predicted values for each model (second test)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(y_test_2, y_pred_lr_2, color='blue')
plt.title("Linear Regression - Test 2")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 2)
plt.scatter(y_test_2, y_pred_ridge_2, color='green')
plt.title("Ridge Regression - Test 2")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 3)
plt.scatter(y_test_2, y_pred_lasso_2, color='red')
plt.title("Lasso Regression - Test 2")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 4)
plt.scatter(y_test_2, y_pred_rf_2, color='purple')
plt.title("Random Forest - Test 2")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.tight_layout()
plt.show()
