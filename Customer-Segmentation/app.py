# =====================================================
# CUSTOMER SEGMENTATION USING PCA + KMEANS CLUSTERING
# =====================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================================
# STEP 1: LOAD DATASET
# =====================================================

df = pd.read_csv("customer_data.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# STEP 2: HANDLE MISSING VALUES
# =====================================================

# Fill numerical missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# =====================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# =====================================================

label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le

print("\nData after Encoding:")
print(df.head())

# =====================================================
# STEP 4: FEATURE SCALING
# =====================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("\nScaled Data Shape:", scaled_data.shape)

# =====================================================
# STEP 5: PCA (DIMENSIONALITY REDUCTION)
# =====================================================

# First determine explained variance

pca_full = PCA()

pca_full.fit(scaled_data)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

plt.figure(figsize=(8,5))
plt.plot(
    range(1, len(cumulative_variance)+1),
    cumulative_variance,
    marker='o'
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.show()

# Use 2 principal components for visualization

pca = PCA(n_components=2)

pca_data = pca.fit_transform(scaled_data)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "\nTotal Variance Explained:",
    sum(pca.explained_variance_ratio_)
)

# =====================================================
# STEP 6: VISUALIZE PCA
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(
    pca_data[:,0],
    pca_data[:,1],
    alpha=0.7
)

plt.title("PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# =====================================================
# STEP 7: ELBOW METHOD
# =====================================================

inertia = []

K = range(1,11)

for k in K:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(pca_data)

    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    K,
    inertia,
    marker='o'
)

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# =====================================================
# STEP 8: SILHOUETTE SCORES
# =====================================================

silhouette_scores = []

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(pca_data)

    score = silhouette_score(
        pca_data,
        labels
    )

    silhouette_scores.append(score)

    print(f"K={k} --> Silhouette Score={score:.4f}")

plt.figure(figsize=(8,5))

plt.plot(
    range(2,11),
    silhouette_scores,
    marker='o'
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.grid(True)
plt.show()

# =====================================================
# STEP 9: FIND BEST K
# =====================================================

best_k = np.argmax(silhouette_scores) + 2

print("\nBest K based on Silhouette Score:", best_k)

# =====================================================
# STEP 10: FINAL KMEANS MODEL
# =====================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(pca_data)

df["Cluster"] = clusters

print("\nCluster Counts:")
print(df["Cluster"].value_counts())

# =====================================================
# STEP 11: CLUSTER VISUALIZATION
# =====================================================

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    pca_data[:,0],
    pca_data[:,1],
    c=clusters,
    cmap="viridis",
    alpha=0.8
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Customer Segments")

plt.colorbar(scatter)

plt.show()

# =====================================================
# STEP 12: CLUSTER PROFILE
# =====================================================

print("\nCluster Profiles:\n")

cluster_profile = (
    df.groupby("Cluster")
      .mean(numeric_only=True)
)

print(cluster_profile)

# Save cluster profile

cluster_profile.to_csv(
    "cluster_profiles.csv"
)

# =====================================================
# STEP 13: CUSTOMER PERSONAS
# =====================================================

print("\n==============================")
print("CUSTOMER PERSONAS")
print("==============================\n")

for cluster in sorted(df["Cluster"].unique()):

    print(f"\nCluster {cluster}")
    print("-" * 30)

    cluster_data = df[df["Cluster"] == cluster]

    print(
        "Customers:",
        len(cluster_data)
    )

    print(
        cluster_data.mean(
            numeric_only=True
        )
        .sort_values(ascending=False)
        .head(5)
    )

# =====================================================
# STEP 14: SAVE FINAL DATASET
# =====================================================

df.to_csv(
    "customer_segmented.csv",
    index=False
)

print("\nFiles Saved Successfully:")
print("1. customer_segmented.csv")
print("2. cluster_profiles.csv")

# =====================================================
# END OF PROJECT
# =====================================================