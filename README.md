# Data Mining Projects

Eren ŞİŞMAN - 201401021

## Project 1

Dataset Selection and Overview

For this assignment, the Palmer Penguins dataset was selected. This dataset is publicly available and designed to classify penguin species (Adelie, Chinstrap, and Gentoo) based on physical measurements and categorical variables. It is well-suited for classification tasks.

Dataset Features:
- Features: bill length, bill depth, flipper length, body mass, and sex.
- Target Variable: species (Adelie, Chinstrap, Gentoo).
The task involves applying a decision tree classifier, comparing it with a random forest classifier, and evaluating their performance with accuracy, precision, recall, and F1 score.

Code Explanation

1. Necessary Libraries

- Libraries used:
  - pandas and numpy for data manipulation.
  - sklearn for machine learning models and performance evaluation.
  - seaborn and matplotlib for visualization.
  - palmerpenguins to load the dataset.

2. Data Loading and Preprocessing

- Data Loading:
  - The dataset is loaded using `load_penguins()` and displayed using `head()` to examine its structure.

- Preprocessing:
  - Feature selection: Only relevant columns are retained.
  - Handling missing values:
    - Numerical columns are filled with the mean value.
    - Missing values in the categorical column `sex` are filled with "Unknown".
  - Encoding categorical variables:
    - `sex` is converted to dummy variables.
    - The target variable `species` is mapped to numerical values.

- Data splitting and scaling:
    - Data is split into training (80%) and test (20%) sets.
    - Numerical features are standardized using `StandardScaler`.

3. Model Training

- Decision Tree:
  - First Training: Default parameters were used.
  - Second Training: Optimized hyperparameters were applied (max_depth=5, min_samples_split=10).
- Random Forest:
  - First Training: Hyperparameters were applied (n_estimators=5, max_depth=3).
  - Second Training: Optimized hyperparameters were applied (n_estimators=200, max_depth=10).

4. Model Evaluation

- A custom function `evaluate_model` computes performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Classification Report

5. Visualization

The confusion matrix provides a clear visualization of model predictions versus actual labels.

### Results

Performance Metrics:

1. Decision Tree:

In the first training, the decision tree performed well but not perfectly. This was due to the default hyperparameters, which allowed some minor misclassifications.

In the second training, hyperparameter optimization ( limiting the depth of the tree and controlling the minimum samples per split) improved the model's ability to generalize. The decision tree achieved more accuracy in this phase, demonstrating the importance of proper parameter tuning.

2.  Random Forest:

In the first training, the random forest model had intentionally limited capacity due to the hyperparameter settings (n_estimators=5 and max_depth=3). This led to underfitting, with lower accuracy and weaker overall performance.

In the second training, increasing the number of estimators (n_estimators=200) and depth (max_depth=10) allowed the random forest to model the data more effectively. This resulted in a significant performance boost, matching the near-optimal behavior expected from a robust ensemble model.

![image](https://github.com/user-attachments/assets/89d7bdf2-187a-4a52-879b-6c5413397ee8)
![image](https://github.com/user-attachments/assets/08cabacf-db89-4ae2-8737-e9375253f799)

Confusion Matrices:

Decision Tree:

In the first training, minor misclassifications were observed, particularly between species with similar physical traits (e.g., Adelie vs. Gentoo).

In the second training, the decision tree correctly classified all samples across all species, as shown in the confusion matrix.

Random Forest:

The first training phase showed more pronounced misclassifications due to the simplified model structure.

The second training phase reduced these errors significantly, resulting in high precision, recall, and F1 scores.

1. Decision Tree - First Training
![image](https://github.com/user-attachments/assets/e68db930-7b6e-4257-a8c5-395bac196996)

2. Random Forest - First Training
![image](https://github.com/user-attachments/assets/3cea142c-09d1-40e3-8cb6-0d2376edc1e7)

3. Decision Tree - Second Training
![image](https://github.com/user-attachments/assets/e1bf74f7-b0f2-4a5d-9204-6102f5f3e8bc)

4. Random Forest - Second Training
![image](https://github.com/user-attachments/assets/8e311ab3-338c-4e71-ba0a-86241f7a32ba)

## Project 2

Dataset Selection and Overview

For this task, we use the mpg dataset (from the ggplot2 package in R), which contains fuel economy data for cars. This dataset is a great choice because it provides continuous numerical values for various attributes like engine displacement, highway mileage, and number of cylinders, which can be used for clustering purposes. The goal of this exercise is to use K-Means clustering to group the data based on the engine displacement (displ) and highway miles per gallon (hwy) features, aiming to predict the cylinder type (cyl).

Dataset features include:

displ: Engine displacement (in liters).

hwy: Highway miles per gallon.

cyl: Number of cylinders.

class: Type of car (e.g., compact, SUV, etc.).

We use K-Means clustering to explore the natural groupings within the dataset.

We chose this dataset because it provides numerical features related to car performance (displacement and highway MPG), and the target variable (cylinder type) is a categorical variable, making it a suitable candidate for clustering tasks.

Code Explanation

1. Importing the Libraries:

-  pandas and numpy: Libraries for data manipulation and numerical computations.

-  train_test_split: Function for splitting the dataset into training and testing sets.

-  KMeans: Clustering algorithm used for unsupervised learning.

-  silhouette_score: Metric used to evaluate the quality of clusters.

-  StandardScaler: Used for standardizing the data (feature scaling).

-  seaborn and matplotlib: Libraries for creating visualizations (scatter plots, elbow method, etc.).

2. Loading and Preprocessing Data:

-  Loading the dataset: Reads the 'mpg.csv' file into a pandas DataFrame.

-  Selecting relevant features: The dataset is reduced to three columns: 'displ' (engine displacement), 'hwy' (highway MPG), and 'cyl' (cylinder type).

-  Handling missing values: Missing values are filled using the most frequent value (mode) of each column.

3. Splitting Data into Training and Testing Sets:

train_test_split: The data is split into training and testing sets, with 70% of the data used for training and 30% for testing. X consists of the feature variables ('displ' and 'hwy'), while y is the target variable ('cyl').

4. Scaling the Data:

The data is standardized using StandardScaler, which centers the data around 0 and scales it to have unit variance. The scaler is fit to the training data and then applied to both training and testing sets.

5. First Test:

-  Model Setup: K-Means is applied with k=2, which is a suboptimal choice for this dataset. We initialize the model with n_clusters=2 and use a fixed random seed for reproducibility.

-  Model Training and Prediction: The model is trained on the scaled training data (X_train_scaled), and predictions are made on the scaled test data (X_test_scaled). The predicted clusters are stored in clusters_bad.

-  Evaluation: The silhouette score is calculated for k=2 using the silhouette_score function. A lower silhouette score indicates that the clustering is not very well-separated.

-  Visualization: The resulting clusters are visualized in a scatter plot, where each data point is colored according to its predicted cluster. The plot uses Engine Displacement (displ) on the x-axis and Highway MPG (hwy) on the y-axis, showing how the data points are distributed into two clusters.

5. Elbow Method for Finding the Optimal Number of Clusters (k):

-  Purpose: To determine the optimal number of clusters (k) for K-Means, we use the Elbow Method. This method involves running the K-Means algorithm for different values of k and plotting the inertia (sum of squared distances from points to their assigned cluster centers).

-  Inertia Calculation: The inertia values are calculated for k ranging from 1 to 10. As k increases, the inertia decreases, but the rate of decrease slows down after a certain point. The "elbow" in the plot indicates the optimal number of clusters.

-  Visualization: The inertia values are plotted against k to visually identify the elbow point. The plot helps us select the optimal number of clusters based on where the inertia reduction slows down.

5. Second Test:

-  Model Setup: Based on the Elbow Method, we apply K-Means with k=3 for optimal clustering. This choice of k aims to create better-defined clusters based on the dataset's distribution.

-  Model Training and Prediction: The K-Means model is trained on the scaled training data (X_train_scaled) and the predictions are made on the test data (X_test_scaled). The predicted clusters are stored in clusters_good.

-  Evaluation: The silhouette score is calculated for k=3 to evaluate the quality of the clustering. A higher silhouette score indicates better-defined clusters.

-  Visualization: The clusters are visualized in a scatter plot, with data points colored based on their predicted clusters. The plot uses Engine Displacement (displ) on the x-axis and Highway MPG (hwy) on the y-axis, showing how the points are now grouped into three clusters.

### Results

![image](https://github.com/user-attachments/assets/d2857490-6886-40d0-b967-8c004666326c)

Test 1: Using K=2

Silhouette Score: The Silhouette score for k=2 was 0.62, which indicates suboptimal clustering, as the number of clusters does not match the true distribution of cylinder types in the dataset.

Test 2: Using K=3 (Optimized Hyperparameters)

Silhouette Score: The Silhouette score for k=3 was 0.47, which is an improvement over the first test. However, the score is still relatively moderate, indicating that there is some overlap in the data clusters, but they are better separated than in Test 1.

Visualizations

1st Test Clustering:
![image](https://github.com/user-attachments/assets/3f1dc2e7-b0cc-41fd-b77b-917b5e327627)

Elbow Method Visualization:
![image](https://github.com/user-attachments/assets/701ee4a8-364a-4892-8527-529318639c06)

2nd Test Clustering:
![image](https://github.com/user-attachments/assets/d2c13221-1a73-4da6-918f-b6bf6c675b83)

- 1st K-Means Clustering (k=2): The first plot shows that the points are divided into two clusters, but the clustering is not meaningful as the dataset should ideally have three distinct clusters.

- Elbow Method: The elbow plot shows a clear bend at k=3, indicating that three clusters is the optimal choice for this dataset.

- 2nd K-Means Clustering (k=3): In the second plot, the data is clustered into three groups, representing the three cylinder types (4, 6, and 8). The separation between the clusters is clearer and more meaningful.

## Project 3

Dataset Selection and Overview

This report section covers the analysis of the Palmer Penguins dataset, where the goal is to predict the body mass of penguins using various regression techniques: Linear Regression, Ridge Regression, Lasso Regression, and Random Forest Regression. The models' performances are evaluated using metrics such as Mean Squared Error (MSE) and R².

Code Expla	nation

1.  Library Imports:

We import necessary libraries like pandas, numpy, and matplotlib for data manipulation, numerical operations, and visualization. We also import sklearn modules for model building, evaluation, and preprocessing, along with seaborn for visualization.

2.  Loading the Dataset:

The Palmer Penguins dataset is loaded using load_penguins() from the palmerpenguins library. This dataset contains data about penguin species, measurements like bill length, flipper length, body mass, and other attributes.

3.  Data Preprocessing:

Selecting Relevant Features: Only the relevant numerical features and the categorical feature sex are selected for analysis: "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", and "sex".

Handling Missing Values: Missing values in the numerical columns are imputed using the most frequent value (mode) via SimpleImputer(strategy="most_frequent"). This ensures that there are no missing values during model training. The sex column is also filled with the mode (the most frequent category).

Converting Categorical Feature: The sex column is converted into numerical variables.

4.  Feature and Target Separation:

The features (X) are separated from the target (y). The target variable body_mass_g is stored in y, while the rest of the columns form the feature set in X.

5.  Train-Test Split:

The data is split into training and test sets using train_test_split().

Test 1: The test size is set to 50%, which means half of the data is used for testing. This creates a more challenging setup with less data for training.

Test 2: The test size is set to 20%, providing a larger training set, allowing the model to learn better patterns and improving the overall model performance.

6.  Feature Scaling:

Standardization: The features are standardized using StandardScaler(), which scales the features to have a mean of 0 and a standard deviation of 1. This helps the models to converge faster and perform better, especially for algorithms like Ridge and Lasso that are sensitive to the scale of the features.

7.  Model Training for Test 1:

Linear Regression: A basic linear regression model is trained without any regularization.

Ridge Regression: Ridge regression is applied with a higher regularization strength (alpha=10) to prevent overfitting.

Lasso Regression: Lasso regression is applied with an alpha=1, which applies L1 regularization and encourages sparse solutions.

Random Forest Regression: A random forest model is trained with parameters (n_estimators=10, max_depth=2)

8.  Model Training for Test 2:

Ridge Regression: The regularization strength is lowered (alpha=0.1) for better model fitting.

Lasso Regression: The alpha for Lasso is reduced to 0.01 to allow more features to remain in the model.

Random Forest Regression: The model complexity is increased by setting n_estimators=500 and max_depth=20, allowing the model to learn better patterns from the data.

Linear Regression: Linear regression remains the same but benefits from the better training set in Test 2.

9.  Model Evaluation:

After fitting the models, predictions are made on the test sets using predict().

Mean Squared Error (MSE) and R² (Coefficient of Determination) are calculated for each model to evaluate the model's performance:

MSE measures the average squared difference between actual and predicted values. Lower MSE indicates a better fit.

R² indicates the proportion of variance in the target variable explained by the model. Higher R² is better, with values closer to 1 indicating a good fit.

10. Visualization:

Scatter plots are generated to compare the actual vs predicted values for each model in both Test 1 and Test 2.

These plots help to visualize how closely the predicted values align with the actual values, with a better-fitting model showing points close to the line y=x.

Models Used

The following models were used to predict the body mass of penguins:

1. Linear Regression: A basic regression model without regularization to observe baseline performance.

2. Ridge Regression: A linear model with L2 regularization to prevent overfitting and improve generalization.

3. Lasso Regression: Similar to Ridge, but with L1 regularization, which also performs feature selection.

4. Random Forest Regression: A non-linear model that aggregates multiple decision trees to make predictions.

### Results

Test 1:
![image](https://github.com/user-attachments/assets/f0846afd-1648-4f7f-82a7-6a49b72834a4)

Test 2 (with optimized performance settings):
![image](https://github.com/user-attachments/assets/4159c459-4da6-4810-b2e8-df32f2a50c92)

Linear Regression: In Test 2, Linear Regression showed a slight improvement in performance with a lower MSE and higher R², indicating a better fit and prediction accuracy compared to Test 1.

Ridge Regression: Ridge regression performed similarly to Linear Regression in both tests, but with Test 2 showing a slight improvement, particularly in MSE, due to reduced regularization (α=0.1).

Lasso Regression: Lasso Regression's performance was almost identical to Linear and Ridge in both tests, with a marginal improvement in Test 2 due to reduced regularization (α=0.01), improving MSE.

Random Forest:  Random Forest showed the greatest improvement in Test 2, achieving the lowest MSE and highest R², thanks to optimized hyperparameters (n_estimators=500, max_depth=20), significantly outperforming the other models.

Visualizations

1st Test:

The following plots show the actual vs predicted values for each model in Test 1.
![image](https://github.com/user-attachments/assets/04bab685-291f-4bcb-956a-d482021e7c98)

2nd Test:

The following plots show the actual vs predicted values for each model in Test 2.
![image](https://github.com/user-attachments/assets/9b80bbd7-e1c4-459d-b48a-3d3c56e90237)

In Test 2, by optimizing the hyperparameters and adjusting the test size, we were able to improve the model's performance. The use of more suitable hyperparameters allowed the models to better capture the underlying patterns in the data, resulting in more accurate predictions and a better overall fit compared to Test 1.
