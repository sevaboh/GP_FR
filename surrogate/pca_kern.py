#!/usr/bin/python
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# === PARAMETERS ===
file_path = sys.argv[1]
n_inputs = 9

# Kernel PCA params
kernel = sys.argv[2]     # 'rbf', 'poly', 'sigmoid', 'cosine'
degree = int(sys.argv[3])
n_components = 2

vis=0
if len(sys.argv)>4:
    vis=int(sys.argv[4])

eps = 0.05          # radius (tune this!)
if len(sys.argv)>5:
    eps=float(sys.argv[5])

# DBSCAN
min_samples = 5    # minimum cluster size

# === LOAD DATA ===
df = pd.read_csv(file_path, sep='\s+', header=None)
ids = df.iloc[:, 0].values
X = df.iloc[:, 1:1+n_inputs]
Y = df.iloc[:, 1+n_inputs:]

Y.columns = [f"y{i+1}" for i in range(Y.shape[1])]

# === REMOVE CONSTANT OUTPUTS ===
tol = 1e-12
Y = Y.loc[:, Y.var() > tol]

# === STANDARDIZE ===
scaler = StandardScaler()
Y_scaled = scaler.fit_transform(Y)

dists = pairwise_distances(Y_scaled)
gamma = 1.0 / (2 * np.median(dists)**2 + 1e-12)

# === KERNEL PCA ===
kpca = KernelPCA(
    n_components=n_components,
    kernel=kernel,
    gamma=gamma,
    degree=degree,
    fit_inverse_transform=False
)

Y_kpca = kpca.fit_transform(Y_scaled)

# === VISUALIZATION ===

if vis==1:
    plt.figure()
    plt.scatter(Y_kpca[:, 0], Y_kpca[:, 1], s=20)
    plt.title(f"Kernel PCA ({kernel}) Projection")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid()
    plt.show()

# === DBSCAN CLUSTERING ===
db = DBSCAN(eps=eps, min_samples=min_samples)
labels = db.fit_predict(Y_kpca)

# === ANALYZE CLUSTERS ===
unique, counts = np.unique(labels, return_counts=True)
cluster_sizes = dict(zip(unique, counts))

# === REMOVE NOISE (-1) ===
valid_mask = labels != -1

if np.sum(valid_mask) == 0:
    print("\nNo clusters found (everything is noise). Try increasing eps.")
    largest_indices = np.array([])
else:
    unique_valid, counts_valid = np.unique(labels[valid_mask], return_counts=True)
    largest_cluster = unique_valid[np.argmax(counts_valid)]

    largest_indices = np.where(labels == largest_cluster)[0]
    largest_ids = ids[largest_indices]


# === VISUALIZATION ===
if vis==1:
    plt.figure()
    scatter = plt.scatter(Y_kpca[:, 0], Y_kpca[:, 1], c=labels)
    plt.title("KPCA + DBSCAN Clustering")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid()
    plt.colorbar(scatter, label="Cluster")
    plt.show()

# === SAVE ORIGINAL ROWS OF LARGEST CLUSTER TO SPACE-SEPARATED TXT ===
if len(largest_indices) > 0:
    output_txt_file = "gp_constants_largest_cluster.txt"
    
    df.iloc[largest_indices].to_csv(
        output_txt_file,
        sep=' ',       # space-separated
        header=False,  # no column names
        index=False
    )

# --- Full dataset correlation ---
input_names = [f"x{i+1}" for i in range(X.shape[1])]
output_names = list(Y.columns)

X.columns = input_names
Y.columns = output_names
full_corr = X.join(Y).corr().loc[input_names, output_names]
print("\nCorrelation matrix (full dataset):")
print(full_corr)

# --- Largest cluster correlation ---
if len(largest_indices) > 0:
    X_cluster = X.iloc[largest_indices]
    Y_cluster = Y.iloc[largest_indices]

    cluster_corr = X_cluster.join(Y_cluster).corr().loc[input_names, output_names]
    print("\nCorrelation matrix (largest cluster, size - "+str(len(largest_indices))+"):")
    print(cluster_corr)
else:
    print("\nNo largest cluster found. Skipping cluster correlation.")
