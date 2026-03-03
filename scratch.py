import pandas as pd
from scipy.stats import mannwhitneyu

king = pd.read_csv("ursus_king.kin0", sep=r"\s+", engine="python")

def evaluate_k(pheno_file, column_name):
    clusters = pd.read_csv(pheno_file, sep="\t")
    clusters = clusters[["IID", column_name]].rename(columns={column_name: "group"})
    
    df = king.copy()
    
    df = df.merge(clusters, left_on="IID1", right_on="IID")
    df = df.rename(columns={"group": "group1"}).drop(columns="IID")
    
    df = df.merge(clusters, left_on="IID2", right_on="IID")
    df = df.rename(columns={"group": "group2"}).drop(columns="IID")
    
    df["relationship"] = df.apply(
        lambda row: "within" if row["group1"] == row["group2"] else "between",
        axis=1
    )
    
    within = df[df["relationship"] == "within"]["KINSHIP"]
    between = df[df["relationship"] == "between"]["KINSHIP"]
    
    stat, p = mannwhitneyu(within, between, alternative="greater")
    
    print(f"\nResults for {column_name}")
    print("Mean within:", within.mean())
    print("Mean between:", between.mean())
    print("Difference:", within.mean() - between.mean())
    print("p-value:", p)

evaluate_k("k2.pheno", "k2")
evaluate_k("k3.pheno", "k3")
evaluate_k("k4.pheno", "k4")