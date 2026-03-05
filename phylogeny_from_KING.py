import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skbio import DistanceMatrix
from skbio.tree import nj
from Bio import Phylo
from matplotlib.lines import Line2D

KING_TABLE = "ursus_king.kin0"
K3_PHENO = "k3.pheno"
OUT_NWK = "ursus_king_nj_tree.nwk"
OUT_PDF = "ursus_tree_k3_KING_colored.pdf"

# -----------------------------
# Build NJ tree from KING
# -----------------------------
king = pd.read_csv(KING_TABLE, sep=r"\s+", engine="python")

ids = sorted(set(king["IID1"]).union(set(king["IID2"])))
idx = {x: i for i, x in enumerate(ids)}

M = np.full((len(ids), len(ids)), np.nan)
for _, r in king.iterrows():
    i = idx[str(r["IID1"]).strip()]
    j = idx[str(r["IID2"]).strip()]
    M[i, j] = r["KINSHIP"]
    M[j, i] = r["KINSHIP"]

kmax = np.nanmax(M)
np.fill_diagonal(M, kmax)

D = kmax - M
D = np.nan_to_num(D, nan=np.nanmax(D))
np.fill_diagonal(D, 0.0)

dm = DistanceMatrix(D, ids)
tree_skbio = nj(dm)

with open(OUT_NWK, "w") as f:
    tree_skbio.write(f)
print(f"Wrote: {OUT_NWK}")

# -----------------------------
# Load tree for plotting
# -----------------------------
tree = Phylo.read(OUT_NWK, "newick")

# -----------------------------
# Load k3 assignments
# -----------------------------
k3 = pd.read_csv(K3_PHENO, sep="\t", engine="python")

# If your file has no header, use this instead:
# k3 = pd.read_csv(K3_PHENO, sep="\t", header=None, names=["FID","IID","k3"])

k3["IID"] = k3["IID"].astype(str).str.strip()
k3["k3"] = k3["k3"].astype(str).str.strip()

# If k3 is numeric (1/2/3), convert to C1/C2/C3
# If it's already C1/C2/C3, this keeps it.
k3["k3_norm"] = k3["k3"].apply(lambda x: f"C{x}" if x.isdigit() else x)

iid_to_group = dict(zip(k3["IID"], k3["k3_norm"]))

cluster_colors = {"C1": "#4C72B0", "C2": "#DD8452", "C3": "#55A868"}

# Build label_colors dict for Biopython draw
tip_labels = [t.name.strip() for t in tree.get_terminals() if t.name]
label_colors = {}
missing = []

for lbl in tip_labels:
    grp = iid_to_group.get(lbl)
    if grp in cluster_colors:
        label_colors[lbl] = cluster_colors[grp]
    else:
        missing.append(lbl)

print(f"Tree tips: {len(tip_labels)}")
print(f"Colored tips: {len(label_colors)}")
print(f"Missing group assignment: {len(missing)}")
if missing:
    print("First 10 missing tip labels:", missing[:10])
    print("First 10 k3 IDs:", list(iid_to_group.keys())[:10])

# -----------------------------
# Plot (label colors + optional colored dots)
# -----------------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1)

Phylo.draw(
    tree,
    axes=ax,
    do_show=False,
    label_func=lambda clade: clade.name if clade.is_terminal() else None,
    label_colors=label_colors,   # <-- key line
)

# Make text readable
for text in ax.texts:
    text.set_fontsize(7)

# Legend
handles = [
    Line2D([0], [0], marker="o", color="w", label=g, markerfacecolor=c, markersize=8)
    for g, c in cluster_colors.items()
]
ax.legend(handles=handles, title="k=3 group", loc="upper right", frameon=True)

ax.set_title("NJ tree from KING-based distances (tips colored by k=3)")
plt.tight_layout()
plt.savefig(OUT_PDF, dpi=300)
plt.show()

print(f"Wrote: {OUT_PDF}")