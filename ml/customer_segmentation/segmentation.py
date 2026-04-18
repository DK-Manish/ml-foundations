import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — writes charts to files instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

df = pd.read_csv("Mall_Customers.csv")

print("=" * 60)
print("          CUSTOMER SEGMENTATION")
print("=" * 60)
print(f"\n   Total customers : {len(df)}")
print(f"   Columns         : {list(df.columns)}")

missing = df.isnull().sum().sum()
print(f"\n   Missing values : {missing}")

df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
print("   Gender encoded : Male=0, Female=1")

df.drop(columns=["CustomerID"], inplace=True)

features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n   Features scaled : mean=0, std=1")

print("\n   Running Elbow Method (K = 1 to 10)...")
inertia_values = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia_values.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia_values, marker="o", color="steelblue", linewidth=2, markersize=7)
plt.title("Elbow Method — Optimal Number of Clusters", fontsize=14)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.xticks(k_range)
plt.axvline(x=5, color="red", linestyle="--", alpha=0.6, label="Chosen K = 5")
plt.legend()
plt.tight_layout()
plt.savefig("elbow_chart.png", dpi=150)
plt.close()
print("   Elbow chart saved  : elbow_chart.png")

K = 5
model = KMeans(n_clusters=K, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

print(f"\n   K-Means trained  : K = {K} clusters")
print(f"   Final inertia    : {model.inertia_:.2f}")

palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette=palette,
    s=80,
    alpha=0.85,
)

# inverse_transform to plot centroids in original units, not scaled space
centroids_scaled   = model.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

for i, (cx, cy) in enumerate(centroids_original):
    plt.scatter(cx, cy, s=250, marker="X", color=palette[i], edgecolors="black", linewidths=1.2, zorder=5)

plt.title("Customer Segments — Annual Income vs Spending Score", fontsize=14)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster", loc="upper left")
plt.tight_layout()
plt.savefig("clusters_income_vs_spending.png", dpi=150)
plt.close()
print("   Cluster chart saved : clusters_income_vs_spending.png")

pair_df = df.copy()
pair_df["Cluster"] = pair_df["Cluster"].astype(str)  # cast to str so seaborn treats it as a category
pair_fig = sns.pairplot(pair_df, hue="Cluster", palette=palette, diag_kind="kde", plot_kws={"alpha": 0.6})
pair_fig.figure.suptitle("Pairplot of All Features by Cluster", y=1.02, fontsize=13)
pair_fig.savefig("pairplot_clusters.png", dpi=150)
plt.close()
print("   Pairplot saved      : pairplot_clusters.png")

print("\n" + "=" * 60)
print("            CLUSTER SUMMARY")
print("=" * 60)

summary = df.groupby("Cluster").agg(
    Count=("Cluster", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
    Pct_Female=("Gender", "mean"),  # mean of 0/1 encodes fraction of females
).round(1)


def label_cluster(row):
    income  = row["Avg_Income"]
    spend   = row["Avg_Spending"]
    mid_inc = df["Annual Income (k$)"].mean()
    mid_sp  = df["Spending Score (1-100)"].mean()

    if income >= mid_inc and spend >= mid_sp:
        return "High Income, High Spenders"
    elif income >= mid_inc and spend < mid_sp:
        return "High Income, Low Spenders"
    elif income < mid_inc and spend >= mid_sp:
        return "Low Income, High Spenders"
    elif income < mid_inc and spend < mid_sp:
        return "Low Income, Low Spenders"
    else:
        return "Average / Mixed"


summary["Segment Label"] = summary.apply(label_cluster, axis=1)

print(f"\n{'Cluster':<10} {'Count':<8} {'Avg Age':<10} {'Avg Income':<14} {'Avg Spend':<12} {'% Female':<12} Segment")
print("-" * 90)
for cluster_id, row in summary.iterrows():
    pct_female = row["Pct_Female"] * 100
    print(
        f"   {cluster_id:<7} {int(row['Count']):<8} {row['Avg_Age']:<10} "
        f"{row['Avg_Income']:<14} {row['Avg_Spending']:<12} {pct_female:<12.1f} {row['Segment Label']}"
    )

results_lines = [
    "=" * 60,
    "          CUSTOMER SEGMENTATION — RESULTS",
    f"          Run date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 60,
    "",
    "DATASET SUMMARY",
    f"   Total customers : {len(df)}",
    f"   Missing values  : {missing}",
    f"   Features used   : Annual Income (k$), Spending Score (1-100)",
    "",
    "MODEL",
    "   Algorithm : K-Means Clustering",
    f"   K (clusters) : {K}",
    f"   Final inertia : {model.inertia_:.2f}",
    "   K chosen via  : Elbow Method",
    "",
    "ELBOW METHOD — Inertia per K",
    f"   {'K':<5} {'Inertia':>10}",
    f"   {'-'*18}",
]

for k, inertia in zip(k_range, inertia_values):
    marker = "  <-- chosen" if k == K else ""
    results_lines.append(f"   {k:<5} {inertia:>10.2f}{marker}")

results_lines += [
    "",
    "=" * 60,
    "CLUSTER SUMMARY",
    "=" * 60,
    f"{'Cluster':<10} {'Count':<8} {'Avg Age':<10} {'Avg Income':<14} {'Avg Spend':<12} {'% Female':<12} Segment",
    "-" * 90,
]

for cluster_id, row in summary.iterrows():
    pct_female = row["Pct_Female"] * 100
    results_lines.append(
        f"   {cluster_id:<7} {int(row['Count']):<8} {row['Avg_Age']:<10} "
        f"{row['Avg_Income']:<14} {row['Avg_Spending']:<12} {pct_female:<12.1f} {row['Segment Label']}"
    )

results_lines += [
    "",
    "CHARTS SAVED",
    "   elbow_chart.png                 — Elbow Method (inertia vs K)",
    "   clusters_income_vs_spending.png — Scatter plot of clusters",
    "   pairplot_clusters.png           — Pairplot of all features by cluster",
    "",
]

with open("results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))

print("\n   Results saved : results.txt")
print("=" * 60)
