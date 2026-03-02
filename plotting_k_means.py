import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
import geopandas as gpd
import numpy as np
import matplotlib.colors as mcolors  # (not strictly needed, but fine)

# --- Read PCA (headered eigenvec) ---
pca = pd.read_csv("59-Ursus_pruned.eigenvec", sep=r"\s+", engine="python")
pca.columns = pca.columns.str.strip().str.replace("\ufeff", "", regex=False)
pca = pca.rename(columns={"#FID": "FID"})

# --- Read metadata (your workflow file is usually TSV even if named .csv) ---
meta = pd.read_csv("59-Ursus_workflow_2.csv", sep=None, engine="python")
meta.columns = meta.columns.str.strip().str.replace("\ufeff", "", regex=False)

# Clean join keys
pca["IID"] = pca["IID"].astype(str).str.strip()
meta["BioSample"] = meta["BioSample"].astype(str).str.strip()

# Merge
merged = pca.merge(meta, left_on="IID", right_on="BioSample", how="inner")
print("MERGED rows:", len(merged))

# Ensure numeric lat/long
merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce")
merged["long"] = pd.to_numeric(merged["long"], errors="coerce")
merged = merged.dropna(subset=["lat", "long"])

# --- Cluster using multiple PCs (recommended for horseshoe) ---
pcs = [f"PC{i}" for i in range(1, 11)]  # PC1..PC10
X = merged[pcs].to_numpy()

# Optional: standardize PCs before clustering
X = StandardScaler().fit_transform(X)

k = 7  # start with 3 for visualization; you can try 2..5 later
km = KMeans(n_clusters=k, random_state=0, n_init=20)
merged["group"] = km.fit_predict(X)  # groups are 1..k
#merged["group"] = km.fit_predict(X) + 1  # groups are 1..k
merged[["FID", "IID", "group"]].to_csv("clusters_k2.txt", sep="\t", index=False, header=False)
print("Wrote: clusters_k3.txt")

merged["k3"] = "C" + (merged["group"]).astype(str)  # C1, C2, C3
merged[["FID", "IID", "k3"]].to_csv("k2.pheno", sep="\t", index=False)
print("Wrote: k3.pheno")

# ============================================================
# Shared color mapping for BOTH PCA plot and Geo map
# ============================================================
groups = sorted(merged["group"].unique())
k = len(groups)  # (recompute just in case)

base_cmap = plt.get_cmap("tab10", k)  # k discrete colors
group_to_color = {g: base_cmap(i) for i, g in enumerate(groups)}

# ============================================================
# --- Plot PCA colored by group (MATCHES MAP COLORS) ---
# ============================================================
pca_colors = merged["group"].map(group_to_color)

plt.figure()
plt.scatter(merged["PC1"], merged["PC2"], c=pca_colors, s=30, edgecolor="none")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title(f"PCA colored by k-means groups (k={k})")

handles = [
    plt.Line2D([0], [0], marker="o", color="w", label=f"{g}",
               markerfacecolor=group_to_color[g], markersize=8)
    for g in groups
]
plt.legend(handles=handles, title="Clusters", loc="best")

plt.savefig(f"PCA_k{k}.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# --- Plot California map colored by SAME group colors ---
# ============================================================
states = gpd.read_file("tl_2023_us_state/tl_2023_us_state.shp")
ca = states[states["NAME"] == "California"]

gdf = gpd.GeoDataFrame(
    merged.copy(),
    geometry=gpd.points_from_xy(merged["long"], merged["lat"]),
    crs="EPSG:4326"
)
gdf["color"] = gdf["group"].map(group_to_color)

fig, ax = plt.subplots(figsize=(8, 10))
ca.boundary.plot(ax=ax, color="black", linewidth=1)

gdf.plot(
    ax=ax,
    color=gdf["color"],     # <- THIS is what ensures it matches the PCA plot
    markersize=55,
    edgecolor="black",
    linewidth=0.3
)

# build a matching legend
for g in groups:
    ax.scatter([], [], c=[group_to_color[g]], label=f"{g}", s=55,
               edgecolors="black", linewidths=0.3)

ax.legend(title=f"Clusters (k = {k})",loc="upper right", bbox_to_anchor=(0.75, 0.95))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_axis_off()
#ax.set_title(f"PCA K-means Clusters (k={k})")

plt.savefig(f"CA_clusters_k{k}.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Determining which k is the best. K = 2 is the best.
# (unchanged from your code)
# ============================================================
pcs = [f"PC{i}" for i in range(1, 11)]   # PC1..PC10
X = merged[pcs].to_numpy()
X = StandardScaler().fit_transform(X)

for k_test in range(2, 8):
    km = KMeans(n_clusters=k_test, random_state=0, n_init=50)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k={k_test} silhouette={score:.3f}")

merged[["IID", "BioSample", "lat", "long", "group"]].to_csv("ursus_groups_for_mapping.csv", index=False)
print("Wrote: ursus_groups_for_mapping.csv")

# (rest of your script continues...)