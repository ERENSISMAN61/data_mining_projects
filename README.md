# palmer_penguins_analysis

RTEU Computer Engineering - Data Mining Homework 

Eren ŞİŞMAN - 201401021

Question 1 
Dataset Selection and Overview
For this assignment, the Palmer Penguins dataset was selected. This dataset is publicly available and designed to classify penguin species (Adelie, Chinstrap, and Gentoo) based on physical measurements and categorical variables. It is well-suited for classification tasks.

Dataset Features:
- Features: bill length, bill depth, flipper length, body mass, and sex.
- Target Variable: species (Adelie, Chinstrap, Gentoo).
The task involves applying a decision tree classifier, comparing it with a random forest classifier, and evaluating their performance with accuracy, precision, recall, and F1 score.

Code Explanation

1. Necessary Libraries
![image](https://github.com/user-attachments/assets/79b2e19d-2d9e-43dd-9611-6d653f4df926)

 
- Libraries used:
  - pandas and numpy for data manipulation.
  - sklearn for machine learning models and performance evaluation.
  - seaborn and matplotlib for visualization.
  - palmerpenguins to load the dataset.

2. Data Loading and Preprocessing
![image](https://github.com/user-attachments/assets/7bb3ea37-17c1-4ff7-8da6-30521028cd58)


 

- Data Loading:
  - The dataset is loaded using `load_penguins()` and displayed using `head()` to examine its structure.
![image](https://github.com/user-attachments/assets/d6e87b17-43a6-44f7-957e-6d884043d0a5)



- Preprocessing:
  - Feature selection: Only relevant columns are retained.
  - Handling missing values:
    - Numerical columns are filled with the mean value.
    - Missing values in the categorical column `sex` are filled with "Unknown".
  - Encoding categorical variables:
    - `sex` is converted to dummy variables.
    - The target variable `species` is mapped to numerical values.
![image](https://github.com/user-attachments/assets/5d3be163-3352-49de-8cce-0f7f577dafe6)


 

  - Data splitting and scaling:
    - Data is split into training (80%) and test (20%) sets.
    - Numerical features are standardized using `StandardScaler`.

3. Model Training

![image](https://github.com/user-attachments/assets/0ebd145a-da17-47f7-9a9c-43b2e8ce681a)



- Decision Tree:
  - First Training: Default parameters were used.
  - Second Training: Optimized hyperparameters were applied (max_depth=5, min_samples_split=10).
- Random Forest:
  - First Training: Hyperparameters were applied (n_estimators=5, max_depth=3).
  - Second Training: Optimized hyperparameters were applied (n_estimators=200, max_depth=10).

4. Model Evaluation
![image](https://github.com/user-attachments/assets/d59aa523-9537-4a9b-8386-9dafe1db9198)


 

- A custom function `evaluate_model` computes performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Classification Report
5. Visualization
![image](https://github.com/user-attachments/assets/4bcdee3a-927f-4b7e-bfb0-58d38ffe35e6)


 

The confusion matrix provides a clear visualization of model predictions versus actual labels.







Results

Performance Metrics:
1. Decision Tree:
•	In the first training, the decision tree performed well but not perfectly. This was due to the default hyperparameters, which allowed some minor misclassifications.
•	In the second training, hyperparameter optimization ( limiting the depth of the tree and controlling the minimum samples per split) improved the model's ability to generalize. The decision tree achieved more accuracy in this phase, demonstrating the importance of proper parameter tuning.
2.  Random Forest:
•	In the first training, the random forest model had intentionally limited capacity due to the hyperparameter settings (n_estimators=5 and max_depth=3). This led to underfitting, with lower accuracy and weaker overall performance.
•	In the second training, increasing the number of estimators (n_estimators=200) and depth (max_depth=10) allowed the random forest to model the data more effectively. This resulted in a significant performance boost, matching the near-optimal behavior expected from a robust ensemble model.
![image](https://github.com/user-attachments/assets/e501cb3f-320e-4466-afba-98fda0492006)
![image](https://github.com/user-attachments/assets/8def8dc6-b0c5-4700-9e8c-ac22c69f318e)



Confusion Matrices:
1.	Decision Tree:
o	In the first training, minor misclassifications were observed, particularly between species with similar physical traits (e.g., Adelie vs. Gentoo).
o	In the second training, the decision tree correctly classified all samples across all species, as shown in the confusion matrix.
2.	Random Forest:
o	The first training phase showed more pronounced misclassifications due to the simplified model structure.
o	The second training phase reduced these errors significantly, resulting in high precision, recall, and F1 scores.




1. Decision Tree - First Training
 ![image](https://github.com/user-attachments/assets/9e867cc3-e3b3-4023-9fa5-a9ebfbaa76a8)


2. Random Forest - First Training
![image](https://github.com/user-attachments/assets/4b551a11-bf62-4884-884a-9f5b4ba0b6a3)

 


3. Decision Tree - Second Training
![image](https://github.com/user-attachments/assets/ad967963-7869-434b-a007-b82d12000237)

 

4. Random Forest - Second Training
 ![image](https://github.com/user-attachments/assets/6c84a2b8-ebf9-494c-b315-eba45b63bc16)





Question 2
Dataset Selection and Overview

For this assignment, we used the Palmer Penguins dataset, a publicly available dataset that includes measurements of penguins.
The dataset contains various features, and the target of our clustering task is to group the penguins into distinct clusters based on their physical attributes.

Dataset Details:
- Features:
  - Bill length (mm)
  - Bill depth (mm)
  - Flipper length (mm)
  - Body mass (g)
- Target: Grouping penguins into clusters based on their measurements.











Code Explanation
![image](https://github.com/user-attachments/assets/60da3eb1-ca9d-487d-a177-54566eed17e2)

 
Step-by-Step Explanation:

1. Loading the Dataset: 
   - The dataset was loaded using `load_penguins()` from the `palmerpenguins` library.
   
2. Data Preprocessing:
   - We selected numerical features (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, and `body_mass_g`).
   - Missing values were handled using `SimpleImputer` with the mean imputation strategy.
   - The features were scaled using `StandardScaler` to ensure that all features contribute equally to the clustering process.

3. First Test:
   - The first test used a fixed value for `k=7` and only two features: `bill_length_mm` and `flipper_length_mm`.
   - The Silhouette Score was calculated to evaluate the quality of clustering, yielding a score of 0.32.
   - We visualized the clusters using a scatter plot of `bill_length_mm` vs `flipper_length_mm`.

4. Second Test:
   - The second test used the Elbow Method to select the optimal value of `k`. The Elbow Method suggested `k=3` as the optimal number of clusters.
   - The Silhouette Score for the second test was 0.44, which indicates improved clustering compared to the first test.
   - We visualized the clusters using all features (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, and `body_mass_g`) in a scatter plot of `bill_length_mm` vs `flipper_length_mm`.

Results

1st Test:
- Silhouette Score: 0.32 (low)
- The clustering in the first test was suboptimal with `k=7` and limited features. The clusters did not clearly separate the penguin species.

2nd Test:
- Silhouette Score: 0.44 (higher)
- By using the Elbow Method to select `k=3`, we achieved better clustering results. The clusters were more distinct, showing clearer separation between penguin species.

Visualizations

1. 1st Test Clustering: The first clustering visualization shows that the penguins were grouped into 7 clusters, but the clustering is not very meaningful, and the groups are not well-separated.
2. Elbow Method Visualization: The Elbow Method plot helps determine the optimal number of clusters, which suggested `k=3` as the best choice based on the "elbow" of the plot.
3. 2nd Test Clustering: The second test, using `k=3`, produced more meaningful and well-separated clusters, clearly differentiating between the penguin species.

1st Test Clustering:
 ![image](https://github.com/user-attachments/assets/8e41c7de-f739-4a7c-9feb-48171cadb836)

Elbow Method Visualization:
 ![image](https://github.com/user-attachments/assets/87307e70-7299-4b75-85e1-9cc19cf96a6c)

2nd Test Clustering:
 ![image](https://github.com/user-attachments/assets/604d62b8-de5f-415f-ae73-2e26bb7c036a)







Question 3

Dataset Selection and Overview

This report section covers the analysis of the Palmer Penguins dataset, where the goal is to predict the body mass of penguins using various regression techniques: Linear Regression, Ridge Regression, Lasso Regression, and Random Forest Regression. The models' performances are evaluated using metrics such as Mean Squared Error (MSE) and R².

Code Explanation
![image](https://github.com/user-attachments/assets/d3dc61a3-6bf3-43f6-8eed-dc9b8f5a00ef)

 
1.  Library Imports:
•	We import necessary libraries like pandas, numpy, and matplotlib for data manipulation, numerical operations, and visualization. We also import sklearn modules for model building, evaluation, and preprocessing, along with seaborn for visualization.
2.  Loading the Dataset:
•	The Palmer Penguins dataset is loaded using load_penguins() from the palmerpenguins library. This dataset contains data about penguin species, measurements like bill length, flipper length, body mass, and other attributes.
3.  Data Preprocessing:
•	Selecting Relevant Features: Only the relevant numerical features and the categorical feature sex are selected for analysis: "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", and "sex".
•	Handling Missing Values: Missing values in the numerical columns are imputed using the most frequent value (mode) via SimpleImputer(strategy="most_frequent"). This ensures that there are no missing values during model training. The sex column is also filled with the mode (the most frequent category).
•	Converting Categorical Feature: The sex column is converted into numerical variables.
4.  Feature and Target Separation:
•	The features (X) are separated from the target (y). The target variable body_mass_g is stored in y, while the rest of the columns form the feature set in X.
5.  Train-Test Split:
•	The data is split into training and test sets using train_test_split().
o	Test 1: The test size is set to 50%, which means half of the data is used for testing. This creates a more challenging setup with less data for training.
o	Test 2: The test size is set to 20%, providing a larger training set, allowing the model to learn better patterns and improving the overall model performance.
6.  Feature Scaling:
•	Standardization: The features are standardized using StandardScaler(), which scales the features to have a mean of 0 and a standard deviation of 1. This helps the models to converge faster and perform better, especially for algorithms like Ridge and Lasso that are sensitive to the scale of the features.
 ![image](https://github.com/user-attachments/assets/0e301e0f-ceef-4560-a966-1a8a5fcea44c)

7.  Model Training for Test 1:
•	Linear Regression: A basic linear regression model is trained without any regularization.
•	Ridge Regression: Ridge regression is applied with a higher regularization strength (alpha=10) to prevent overfitting.
•	Lasso Regression: Lasso regression is applied with an alpha=1, which applies L1 regularization and encourages sparse solutions.
•	Random Forest Regression: A random forest model is trained with parameters (n_estimators=10, max_depth=2) 


8.  Model Training for Test 2:
•	Ridge Regression: The regularization strength is lowered (alpha=0.1) for better model fitting.
•	Lasso Regression: The alpha for Lasso is reduced to 0.01 to allow more features to remain in the model.
•	Random Forest Regression: The model complexity is increased by setting n_estimators=500 and max_depth=20, allowing the model to learn better patterns from the data.
•	Linear Regression: Linear regression remains the same but benefits from the better training set in Test 2.
9.  Model Evaluation:
•	After fitting the models, predictions are made on the test sets using predict().
•	Mean Squared Error (MSE) and R² (Coefficient of Determination) are calculated for each model to evaluate the model's performance:
o	MSE measures the average squared difference between actual and predicted values. Lower MSE indicates a better fit.
o	R² indicates the proportion of variance in the target variable explained by the model. Higher R² is better, with values closer to 1 indicating a good fit.
 ![image](https://github.com/user-attachments/assets/5651893a-1c7b-4083-86da-d2a7fa8f7582)

10. Visualization:
•	Scatter plots are generated to compare the actual vs predicted values for each model in both Test 1 and Test 2.
o	These plots help to visualize how closely the predicted values align with the actual values, with a better-fitting model showing points close to the line y=x.
Models Used
The following models were used to predict the body mass of penguins:
1. Linear Regression: A basic regression model without regularization to observe baseline performance.
2. Ridge Regression: A linear model with L2 regularization to prevent overfitting and improve generalization.
3. Lasso Regression: Similar to Ridge, but with L1 regularization, which also performs feature selection.
4. Random Forest Regression: A non-linear model that aggregates multiple decision trees to make predictions.
Results

Test 1:
 ![image](https://github.com/user-attachments/assets/b4873c8e-72d0-459c-8aa1-c437b0f817a5)


Test 2 (with optimized performance settings):
 ![image](https://github.com/user-attachments/assets/8222a80c-7fb3-43c8-98a1-b0906114fe85)

	Linear Regression: In Test 2, Linear Regression showed a slight improvement in performance with a lower MSE and higher R², indicating a better fit and prediction accuracy compared to Test 1.
	Ridge Regression: Ridge regression performed similarly to Linear Regression in both tests, but with Test 2 showing a slight improvement, particularly in MSE, due to reduced regularization (α=0.1).
	Lasso Regression: Lasso Regression's performance was almost identical to Linear and Ridge in both tests, with a marginal improvement in Test 2 due to reduced regularization (α=0.01), improving MSE.
	Random Forest:  Random Forest showed the greatest improvement in Test 2, achieving the lowest MSE and highest R², thanks to optimized hyperparameters (n_estimators=500, max_depth=20), significantly outperforming the other models.

Visualizations
1st Test:
The following plots show the actual vs predicted values for each model in Test 1.
 ![image](https://github.com/user-attachments/assets/ee2bd36d-bb24-4717-be0a-6048bf47fb0c)

2nd Test:
The following plots show the actual vs predicted values for each model in Test 2.
 ![image](https://github.com/user-attachments/assets/cddde2fb-1b73-4f96-a50a-560f200d0baa)

In Test 2, by optimizing the hyperparameters and adjusting the test size, we were able to improve the model's performance. The use of more suitable hyperparameters allowed the models to better capture the underlying patterns in the data, resulting in more accurate predictions and a better overall fit compared to Test 1.
