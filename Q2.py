# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins

# load the dataset
penguins = load_penguins()

# select numerical features
penguins = penguins[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]

# handle missing values by filling with mean
imputer = SimpleImputer(strategy="mean")
penguins = pd.DataFrame(imputer.fit_transform(penguins), columns=penguins.columns)

# scale the features to standardize the dataset
scaler = StandardScaler()
penguins_scaled = scaler.fit_transform(penguins)

# first test
kmeans_test1 = KMeans(n_clusters=7, random_state=42)
clusters_test1 = kmeans_test1.fit_predict(penguins_scaled)  
silhouette_test1 = silhouette_score(penguins_scaled, clusters_test1)
print(f"1st test silhouette score: {silhouette_test1:.2f}")

# visualize clusters for the first test
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=penguins["bill_length_mm"],
    y=penguins["flipper_length_mm"], 
    hue=clusters_test1, 
    palette="Set1", 
    style=clusters_test1
)
plt.title("1st test: clustering")
plt.xlabel("bill length (mm)")
plt.ylabel("flipper length (mm)")
plt.legend(title="cluster")
plt.show()

# second test: using elbow method and all features
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(penguins_scaled)
    inertia.append(kmeans.inertia_)

# visualize the elbow method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker="o")
plt.title("elbow method: optimal k selection")
plt.xlabel("number of clusters (k)")
plt.ylabel("inertia")
plt.show()

# optimal k selected as 3
kmeans_test2 = KMeans(n_clusters=3, random_state=42)
clusters_test2 = kmeans_test2.fit_predict(penguins_scaled)
silhouette_test2 = silhouette_score(penguins_scaled, clusters_test2)
print(f"2nd test silhouette score: {silhouette_test2:.2f}")

# visualize clusters for the second test
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=penguins["bill_length_mm"],
    y=penguins["flipper_length_mm"],
    hue=clusters_test2,
    palette="Set2",
    style=clusters_test2,
)
plt.title("2nd test: improved clustering")
plt.xlabel("bill length (mm)")
plt.ylabel("flipper length (mm)")
plt.legend(title="cluster")
plt.show()
