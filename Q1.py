# necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# import the dataset
from palmerpenguins import load_penguins

# load the dataset
penguins = load_penguins()
print(penguins.head())

# select the necessary columns
penguins = penguins[["species", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]]

# check for missing values
imputer = SimpleImputer(strategy="mean")
penguins.iloc[:, 1:-1] = imputer.fit_transform(penguins.iloc[:, 1:-1])

# fill missing
penguins["sex"].fillna("Unknown", inplace=True)

# convert categorical variables to numerical
penguins = pd.get_dummies(penguins, columns=["sex"], drop_first=True)
penguins["species"] = penguins["species"].map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})

# split the data into features and target
X = penguins.drop(columns=["species"])
y = penguins["species"]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# first model training
# decision tree model
dt_model_1 = DecisionTreeClassifier(random_state=42)
dt_model_1.fit(X_train, y_train)
y_pred_dt_1 = dt_model_1.predict(X_test)

# random forest model
rf_model_1 = RandomForestClassifier( n_estimators=5, max_depth=3 ,random_state=42)
rf_model_1.fit(X_train, y_train)
y_pred_rf_1 = rf_model_1.predict(X_test)

# second model training
# decision tree model
dt_model_2 = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_model_2.fit(X_train, y_train)
y_pred_dt_2 = dt_model_2.predict(X_test)

# random forest model
rf_model_2 = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf_model_2.fit(X_train, y_train)
y_pred_rf_2 = rf_model_2.predict(X_test)

# evaluate model performance
def evaluate_model(y_test, y_pred, model_name, training_phase):
    print(f"{model_name} - {training_phase} training")
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("precision:", precision_score(y_test, y_pred, average='weighted'))
    print("recall:", recall_score(y_test, y_pred, average='weighted'))
    print("f1 score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nclassification report:\n", classification_report(y_test, y_pred))

# evaluate the models
evaluate_model(y_test, y_pred_dt_1, "decision tree", "first")
evaluate_model(y_test, y_pred_rf_1, "random forest", "first")
evaluate_model(y_test, y_pred_dt_2, "decision tree", "second")
evaluate_model(y_test, y_pred_rf_2, "random forest", "second")

# confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_name, training_phase):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Adelie", "Chinstrap", "Gentoo"], yticklabels=["Adelie", "Chinstrap", "Gentoo"])
    plt.title(f"{model_name} - {training_phase} training")
    plt.ylabel("actual")
    plt.xlabel("predicted")
    plt.show()

# plot confusion matrix
plot_confusion_matrix(y_test, y_pred_dt_1, "decision tree", "first")
plot_confusion_matrix(y_test, y_pred_rf_1, "random forest", "first")
plot_confusion_matrix(y_test, y_pred_dt_2, "decision tree", "second")
plot_confusion_matrix(y_test, y_pred_rf_2, "random forest", "second")
