# Data Mining Projects

Author: Eren ŞİŞMAN

## Project 1

### Dataset Selection and Overview

For this assignment, the Palmer Penguins dataset was selected. This dataset is publicly available and designed to classify penguin species (Adelie, Chinstrap, and Gentoo) based on physical measurements and categorical variables. It is well-suited for classification tasks.

Dataset Features:
- Features: bill length, bill depth, flipper length, body mass, and sex.
- Target Variable: species (Adelie, Chinstrap, Gentoo).
The task involves applying a decision tree classifier, comparing it with a random forest classifier, and evaluating their performance with accuracy, precision, recall, and F1 score.

Why This Dataset?
The Palmer Penguins dataset was chosen because:
1.	Multi-Class Problem: It has three species (Adelie, Chinstrap, Gentoo), ideal for classification tasks.
2.	Feature Variety: Includes both numerical (e.g., bill_length_mm, body_mass_g) and categorical features (sex).
3.	Balanced Classes: Species are well-distributed, preventing bias in model performance.
4.	Real-World Relevance: Features like bill dimensions and body mass are meaningful in ecological studies.
5.	Manageable Size: Small and easy to preprocess, making it ideal for quick analysis.

The Goal
The goal of this task is to classify penguin species based on physical measurements and demographic features using Decision Tree and Random Forest models. By evaluating and comparing these models, we aim to understand how feature interactions and hyperparameter tuning impact classification accuracy and overall performance.

### Code Explanation

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

### Dataset Selection and Overview

For this task, we use the mpg dataset (from the ggplot2 package in R), which contains fuel economy data for cars. This dataset is a great choice because it provides continuous numerical values for various attributes like engine displacement, highway mileage, and number of cylinders, which can be used for clustering purposes. The goal of this exercise is to use K-Means clustering to group the data based on the engine displacement (displ) and highway miles per gallon (hwy) features, aiming to predict the cylinder type (cyl).

Dataset features include:

displ: Engine displacement (in liters).

hwy: Highway miles per gallon.

cyl: Number of cylinders.

class: Type of car (e.g., compact, SUV, etc.).

We use K-Means clustering to explore the natural groupings within the dataset.

We chose this dataset because it provides numerical features related to car performance (displacement and highway MPG), and the target variable (cylinder type) is a categorical variable, making it a suitable candidate for clustering tasks.

### Code Explanation

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

### Visualizations

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

### Dataset Selection and Overview

For this question, we use the diamonds dataset (from the seaborn library in Python), which contains detailed information about diamonds, including various numerical and categorical attributes. This dataset is an excellent choice because it provides a continuous numerical target variable (price) and other features that can be used to predict diamond prices through regression models.

The diamonds dataset contains the following features:
1.	carat: The weight of the diamond (continuous numerical variable).
2.	cut: Quality of the cut of the diamond (categorical variable, e.g., Fair, Good, Very Good, Premium, Ideal).
3.	color: Diamond color grade, from D (best) to J (worst) (categorical variable).
4.	clarity: Measure of how clear the diamond is, from I1 (lowest clarity) to IF (highest clarity) (categorical variable).
5.	depth: Total depth percentage (continuous numerical variable).
6.	table: Width of the top of the diamond relative to the widest point (continuous numerical variable).
7.	price: Price of the diamond in US dollars (continuous numerical target variable).
8.	x: Length of the diamond in mm (continuous numerical variable).
9.	y: Width of the diamond in mm (continuous numerical variable).
10.	z: Depth of the diamond in mm (continuous numerical variable).


Why This Dataset
We selected this dataset for the following reasons:
- Continuous Target Variable: The price variable provides a continuous numerical target, making it ideal for regression tasks.
- Variety of Features: The dataset includes a mix of categorical and numerical variables, allowing us to demonstrate preprocessing steps such as encoding categorical features and scaling numerical features.
- Real-World Relevance: Predicting diamond prices based on their characteristics is a practical problem with applications in the jewelry industry.
- Complexity and Diversity: The combination of physical dimensions, quality grades, and numerical measurements makes this dataset challenging and engaging for model development and analysis.
The primary objective is to predict the price of diamonds using regression techniques and compare the performance of different models (Linear Regression, Ridge, Lasso, and Random Forest).

### Code Explanation

1.  Library Imports:
 
-  pandas: Handles data manipulation and analysis.
-  train_test_split: Splits data into training and testing sets.
-  Regression Models:
- LinearRegression, Ridge, Lasso (linear models) and RandomForestRegressor (non-linear model) for building regression models.
-  Evaluation Metrics:
- mean_squared_error and r2_score evaluate model performance.
-  StandardScaler: Standardizes features by scaling them to unit variance.
-  SimpleImputer: Handles missing values by filling them with statistical measures.
-  seaborn: Creates appealing data visualizations.
-  matplotlib.pyplot: Plots actual vs. predicted values.

2. Loading and Preprocessing the Dataset:
 

-  The diamonds dataset is loaded, which contains categorical and numerical variables.
-  Columns like cut, color, and clarity are encoded into numeric values using one-hot encoding.
-  Missing values, if any, are handled using the most frequent value (mode).
-  The independent variables (X) and the target variable (y) are defined for regression.



3.  Test 1:
 
-  Data Splitting:
- The dataset is split into 40% training data and 60% testing data.
- The train_test_split function ensures randomness and reproducibility (random_state=42).
-  Feature Scaling:
- Standardization is performed using StandardScaler to ensure all features have a mean of 0 and a standard deviation of 1.
-This step is especially important for models like Ridge and Lasso regression that are sensitive to the scale of features.


-  Model Training:
- Linear Regression:
- A simple linear model is fitted to predict the target variable (price) using scaled training data.
- Ridge Regression:
- Includes L2 regularization to penalize large coefficients, helping to prevent overfitting.
- Default alpha value of 1.0 is used (no hyperparameter tuning).
- Lasso Regression:
 - Uses L1 regularization, which can shrink some coefficients to zero, effectively performing feature selection.
 - Default alpha value of 1.0 is used.
- Random Forest Regression:
 - Ensemble-based model that uses multiple decision trees.
 - Default settings (100 estimators) are used, with no hyperparameter tuning.
-  Prediction:
 - Each model predicts the target variable (price) for the test dataset.
4.  Test 2:
 
-  Data Splitting:
- The dataset is split into 70% training data and 30% testing data to provide more data for training and model optimization.
-   Model Training:
- Ridge Regression:
- Alpha is reduced to 0.1, optimizing the trade-off between bias and variance.
- Lasso Regression:
- Alpha is set to 0.01, enabling better feature selection while maintaining generalization.
- Random Forest Regression:
- The number of trees is increased to 200, and the maximum depth of trees is limited to 20.
- These optimizations aim to improve model performance and prevent overfitting.
-  Prediction:
- Predictions for the target variable (price) are generated for the test dataset.

5.  Evaluation of Models:
 

-  Evaluation Metrics:
- Mean Squared Error (MSE):
- Measures the average squared difference between predicted and actual values.
- Lower MSE indicates better model performance.
- R² Score:
- Indicates the proportion of variance in the target variable explained by the model.
- Higher values indicate a better fit.
-  Model Comparison:
- Both tests evaluate performance across all four models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regression
- Results are printed for Test 1 and Test 2 separately.

6.  Visualization:
  
-  Scatter Plots:
- Each subplot shows the actual values (x-axis) vs predicted values (y-axis).
- Separate visualizations are created for each model in both tests.
-  Analysis:
- Scatter plots reveal the quality of predictions.
- Points closer to the diagonal line (x=y) indicate better predictions.


### Results

Test 1:
 ![image](https://github.com/user-attachments/assets/3344063f-e54b-435d-85d9-a129f029dca5)

Test 2 (with optimized performance settings):
 ![image](https://github.com/user-attachments/assets/e9ecd634-da63-49b6-a96a-5c44efa99c48)

-  Linear Regression:
In Test 2, Linear Regression demonstrated a slight improvement in performance with a reduced MSE and marginally increased R². The improvement was due to the larger training set, enabling the model to better capture the linear relationships in the data. However, it remained limited in capturing any non-linear patterns.
-  Ridge Regression:
Ridge Regression's performance was consistent with Linear Regression across both tests. In Test 2, the model slightly improved, particularly in MSE, due to a reduced regularization parameter (α=0.1), which allowed it to balance bias and variance more effectively without over-penalizing coefficients.
-  Lasso Regression:
Lasso Regression performed similarly to Ridge and Linear Regression, with minor improvements in Test 2. The reduced regularization parameter (α=0.01) in Test 2 allowed the model to better capture significant features, slightly reducing MSE while maintaining simplicity through feature selection.
-  Random Forest Regression:
Random Forest achieved the most significant improvement in Test 2, delivering the lowest MSE and highest R² among all models. This improvement was due to hyperparameter optimization (n_estimators=200, max_depth=20), which allowed it to better capture complex non-linear relationships and interactions between features. It significantly outperformed all linear models in both tests.

### Visualizations
1st Test:
The following plots show the actual vs predicted values for each model in Test 1.
 ![image](https://github.com/user-attachments/assets/913d996c-d62d-4f61-8bfb-0b454ac28777)

2nd Test:
The following plots show the actual vs predicted values for each model in Test 2.
 ![image](https://github.com/user-attachments/assets/d1e10c33-ffff-4184-acc2-19b19e6a3246)

In Test 2, optimizing hyperparameters and using a larger training set significantly improved model performance. The adjustments allowed the models to better capture the underlying relationships within the data, resulting in more accurate predictions and a stronger fit compared to Test 1. These enhancements highlight the importance of proper parameter tuning and dataset splitting in achieving optimal results in regression tasks.
