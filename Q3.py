import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# load the diamonds dataset
diamonds = sns.load_dataset('diamonds')

# data preprocessing
# select relevant features (independent variables) and target variable
diamonds_selected = diamonds[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']]

# handle missing values by filling with the most frequent value (mode) for categorical variables
imputer = SimpleImputer(strategy='most_frequent')  # impute missing values
diamonds_selected = pd.DataFrame(imputer.fit_transform(diamonds_selected), columns=diamonds_selected.columns)

# encode categorical variables (cut, color, clarity) using one-hot encoding
diamonds_selected = pd.get_dummies(diamonds_selected, columns=['cut', 'color', 'clarity'], drop_first=True)

# define features (X) and target variable (y)
X = diamonds_selected.drop(columns=['price'])  # independent variables
y = diamonds_selected['price']  # target variable (price)

# 1st test: train (40%) / test (60%)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.6, random_state=42)

# scale the features to standardize the dataset
scaler = StandardScaler()
X_train_scaled_1 = scaler.fit_transform(X_train_1)
X_test_scaled_1 = scaler.transform(X_test_1)

# initialize and fit the linear regression model for the first test
linear_reg_1 = LinearRegression()
linear_reg_1.fit(X_train_scaled_1, y_train_1)

# make predictions on the test set
y_pred_1 = linear_reg_1.predict(X_test_scaled_1)

# initialize and fit ridge regression for the first test (no hyperparameter tuning)
ridge_reg_1 = Ridge(alpha=1.0) 
ridge_reg_1.fit(X_train_scaled_1, y_train_1)

# make predictions on the test set
y_pred_ridge_1 = ridge_reg_1.predict(X_test_scaled_1)

# initialize and fit Lasso regression for the first test (no hyperparameter tuning)
lasso_reg_1 = Lasso(alpha=1.0) 
lasso_reg_1.fit(X_train_scaled_1, y_train_1)

# make predictions on the test set
y_pred_lasso_1 = lasso_reg_1.predict(X_test_scaled_1)

# initialize and fit random forest regression for the first test (no hyperparameter tuning)
rf_reg_1 = RandomForestRegressor(n_estimators=100, random_state=42)  # default hyperparameters
rf_reg_1.fit(X_train_scaled_1, y_train_1)

# make predictions on the test set
y_pred_rf_1 = rf_reg_1.predict(X_test_scaled_1)

# results for 1st test: evaluate performance for all models
def evaluate_model(y_test, y_pred, model_name, test_number):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - Test {test_number} - MSE: {mse:.2f}, R^2: {r2:.2f}")

# evaluate models for the first test
print("\nResults for Test 1:")
evaluate_model(y_test_1, y_pred_1, "Linear Regression", 1)
evaluate_model(y_test_1, y_pred_ridge_1, "Ridge Regression", 1)
evaluate_model(y_test_1, y_pred_lasso_1, "Lasso Regression", 1)
evaluate_model(y_test_1, y_pred_rf_1, "Random Forest Regression", 1)

# 2nd test: train (70%) / test (30%)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.3, random_state=42)

# scale the features to standardize the dataset
X_train_scaled_2 = scaler.fit_transform(X_train_2)
X_test_scaled_2 = scaler.transform(X_test_2)

# 2nd test: linear regression (no hyperparameter tuning)
linear_reg_2 = LinearRegression()
linear_reg_2.fit(X_train_scaled_2, y_train_2)
y_pred_2 = linear_reg_2.predict(X_test_scaled_2)

# 2nd test: ridge regression with hyperparameter tuning
ridge_reg_2 = Ridge(alpha=0.1)  
ridge_reg_2.fit(X_train_scaled_2, y_train_2)
y_pred_ridge_2 = ridge_reg_2.predict(X_test_scaled_2)

# 2nd test: lasso regression with hyperparameter tuning
lasso_reg_2 = Lasso(alpha=0.01)  
lasso_reg_2.fit(X_train_scaled_2, y_train_2)
y_pred_lasso_2 = lasso_reg_2.predict(X_test_scaled_2)

# 2nd test: random forest with hyperparameter tuning (n_estimators=200, max_depth=20)
rf_reg_2 = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_reg_2.fit(X_train_scaled_2, y_train_2)
y_pred_rf_2 = rf_reg_2.predict(X_test_scaled_2)

# results for 2nd test: evaluate performance for all models
print("\nResults for Test 2:")
evaluate_model(y_test_2, y_pred_2, "Linear Regression", 2)
evaluate_model(y_test_2, y_pred_ridge_2, "Ridge Regression", 2)
evaluate_model(y_test_2, y_pred_lasso_2, "Lasso Regression", 2)
evaluate_model(y_test_2, y_pred_rf_2, "Random Forest Regression", 2)

# visualize the actual vs predicted values for each model (first test)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(y_test_1, y_pred_1, color='blue', alpha = 0.1)
plt.title("Linear Regression - Test 1")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 2)
plt.scatter(y_test_1, y_pred_ridge_1, color='green', alpha = 0.1)
plt.title("Ridge Regression - Test 1")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 3)
plt.scatter(y_test_1, y_pred_lasso_1, color='red' , alpha = 0.1)
plt.title("Lasso Regression - Test 1")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 4)
plt.scatter(y_test_1, y_pred_rf_1, color='purple', alpha = 0.1)
plt.title("Random Forest - Test 1")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.tight_layout()
plt.show()

# Visualize the actual vs predicted values for each model (second test)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(y_test_2, y_pred_2, color='blue', alpha = 0.1)
plt.title("Linear Regression - Test 2")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 2)
plt.scatter(y_test_2, y_pred_ridge_2, color='green', alpha = 0.1)
plt.title("Ridge Regression - Test 2")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 3)
plt.scatter(y_test_2, y_pred_lasso_2, color='red', alpha = 0.1)
plt.title("Lasso Regression - Test 2")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.subplot(2, 2, 4)
plt.scatter(y_test_2, y_pred_rf_2, color='purple', alpha = 0.1)
plt.title("Random Forest - Test 2")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.tight_layout()
plt.show()
