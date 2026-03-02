import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Load CA shapefile
states = gpd.read_file("tl_2023_us_state/tl_2023_us_state.shp")

# Filter to California
ca = states[states["NAME"] == "California"]

# Load coordinates + kinship
king = pd.read_csv("ursus_king.kin0", sep=r"\s+", engine="python")
coords = pd.read_csv("59-Ursus_workflow_2.csv")
coords = coords.rename(columns={"BioSample": "IID"})
coords = coords[["IID", "lat", "long"]]

ref = "CCGPMC010_B2515"

# Extract kinship to reference
a = king[king["IID1"] == ref][["IID2", "KINSHIP"]].rename(columns={"IID2": "IID"})
b = king[king["IID2"] == ref][["IID1", "KINSHIP"]].rename(columns={"IID1": "IID"})
rel = pd.concat([a, b], ignore_index=True)
rel = rel.groupby("IID", as_index=False)["KINSHIP"].mean()

rel_map = coords.merge(rel, on="IID", how="left")

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    rel_map,
    geometry=gpd.points_from_xy(rel_map.long, rel_map.lat),
    crs="EPSG:4326"
)

# Plot
fig, ax = plt.subplots(figsize=(8, 10))

ca.boundary.plot(ax=ax, color="black", linewidth=1)
gdf["rel_strength"] = -gdf["KINSHIP"]
gdf.plot(
    ax=ax,
    column="KINSHIP",
    cmap="Blues",
    markersize=60,
    edgecolor="black",
    legend=True,
        legend_kwds={
        "shrink": 0.6,      # makes it shorter
        "aspect": 20,       # makes it thinner
        "pad": 0.02,        # spacing from plot
    }
)

# Highlight reference bear
ref_point = gdf[gdf["IID"] == ref]
ref_point.plot(
    ax=ax,
    marker="*",
    color="yellow",
    markersize=300,
    edgecolor="black",
    label="Reference Bear"
)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
plt.title(f"Spatial Relatedness to Reference: {ref}")
plt.legend()
plt.savefig(f"king_reference_map_{ref}.png", dpi=600, bbox_inches="tight")
plt.show()