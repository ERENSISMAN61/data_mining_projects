import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# load the mpg dataset
mpg = pd.read_csv('mpg.csv')

# data preprocessing
mpg_selected = mpg[['displ', 'hwy', 'cyl']]

# handle missing values by filling with the most frequent value
mpg_selected = mpg_selected.fillna(mpg_selected.mode().iloc[0])

# split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(mpg_selected[['displ', 'hwy']], mpg_selected['cyl'], test_size=0.3, random_state=42)

# scale the data for clustering (only displ and hwy)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 1st test: using K-Means with k=2 
kmeans_bad = KMeans(n_clusters=2, random_state=42)
kmeans_bad.fit(X_train_scaled)
clusters_bad = kmeans_bad.predict(X_test_scaled)

# evaluate clustering using silhouette score
silhouette_bad = silhouette_score(X_test_scaled, clusters_bad)
print(f"Silhouette score for K=2: {silhouette_bad:.2f}")

# visualize the clusters for the first test (k=2)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['displ'], y=X_test['hwy'], hue=clusters_bad, palette='Set1')
plt.title("K-Means Clustering (k=2) with Cylinder Types")
plt.xlabel("Engine Displacement (displ)")
plt.ylabel("Highway MPG (hwy)")
plt.legend(title="Cylinder Type")
plt.show()

# elbow method for finding the optimal number of clusters (k)
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    inertia.append(kmeans.inertia_)

# plot the Elbow Method to visualize the optimal k
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# based on the elbow plot, we can decide on k=3 as the optimal k.

# 2nd test: using k-means with k=3 (optimized hyperparameters)
kmeans_good = KMeans(n_clusters=3, random_state=42)
kmeans_good.fit(X_train_scaled)
clusters_good = kmeans_good.predict(X_test_scaled)

# evaluate clustering using silhouette score for the good model
silhouette_good = silhouette_score(X_test_scaled, clusters_good)
print(f"Silhouette score for K=3 (optimized hyperparameters): {silhouette_good:.2f}")

# visualize the clusters for the second test (k=3) 
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['displ'], y=X_test['hwy'], hue=clusters_good, palette='Set2')
plt.title("K-Means Clustering (k=3) with Cylinder Types (Optimized Hyperparameters)")
plt.xlabel("Engine Displacement (displ)")
plt.ylabel("Highway MPG (hwy)")
plt.legend(title="Cylinder Type")
plt.show()
