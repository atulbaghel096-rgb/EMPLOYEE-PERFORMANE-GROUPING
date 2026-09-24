import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
data = {
    "Employee": [
        "E1", "E2", "E3", "E4", "E5",
        "E6", "E7", "E8", "E9", "E10",
        "E11", "E12", "E13", "E14", "E15"
    ],
    "Experience": [
        1, 2, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 15
    ],
    "Salary": [
        25000, 28000, 30000, 33000, 36000,
        42000, 45000, 50000, 55000, 60000,
        65000, 70000, 75000, 82000, 90000
    ],
    "Performance_Score": [
        45, 48, 52, 55, 60,
        65, 68, 72, 76, 80,
        83, 86, 89, 92, 96
    ],
    "Working_Hours": [
        35, 36, 37, 38, 39,
        40, 41, 42, 42, 43,
        44, 44, 45, 45, 46
    ]
}

df = pd.DataFrame(data)
print("\nEmployee Dataset:")
print(df)

features = [
    "Experience",
    "Salary",
    "Performance_Score",
    "Working_Hours"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(12, 6))

dendrogram(
    linked,
    labels=df["Employee"].values,
    leaf_rotation=45
)

plt.title("Employee Hierarchical Clustering")
plt.xlabel("Employees")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

df["Cluster"] = fcluster(
    linked,
    3,
    criterion="maxclust"
)

print("\nEmployee Groups:")
print(
    df[
        [
            "Employee",
            "Experience",
            "Salary",
            "Performance_Score",
            "Working_Hours",
            "Cluster"
        ]
    ]
)

summary = df.groupby("Cluster")[features].mean()

print("\nCluster Summary:")
print(summary)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Experience",
    y="Performance_Score",
    hue="Cluster",
    palette="viridis",
    s=150
)

plt.title("Employee Performance Groups")
plt.xlabel("Experience")
plt.ylabel("Performance Score")
plt.show()

for cluster in sorted(df["Cluster"].unique()):
    print(f"\nGroup {cluster}:")
    employees = df[df["Cluster"] == cluster]["Employee"]
    print(list(employees))