import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


data = {
    "Employee": [
        "Amit", "Rahul", "Priya", "Neha", "Vikas",
        "Rohit", "Sneha", "Karan", "Pooja", "Arjun",
        "Simran", "Manish", "Anjali", "Deepak", "Riya"
    ],

    "Experience": [
        2, 5, 3, 8, 10,
        1, 6, 4, 9, 2,
        7, 5, 3, 11, 6
    ],

    "Salary": [
        30000, 55000, 40000, 85000, 95000,
        28000, 65000, 45000, 90000, 32000,
        75000, 60000, 38000, 105000, 70000
    ],

    "Performance_Score": [
        55, 78, 65, 90, 95,
        50, 82, 70, 92, 58,
        88, 80, 68, 96, 85
    ],

    "Working_Hours": [
        38, 42, 40, 45, 44,
        37, 43, 41, 46, 39,
        44, 42, 40, 45, 43
    ]
}

df = pd.DataFrame(data)

print("\nEmployee Dataset:")
print(df)


print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


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
    leaf_rotation=45,
    leaf_font_size=10
)

plt.title("Employee Hierarchical Clustering Dendrogram")
plt.xlabel("Employees")
plt.ylabel("Distance")

plt.tight_layout()
plt.show()

model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

df["Cluster"] = model.fit_predict(X_scaled)

print("\nEmployees with Cluster:")
print(df.sort_values("Cluster"))

cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster Summary:")
print(cluster_summary.round(2))


cluster_performance = df.groupby(
    "Cluster"
)["Performance_Score"].mean()

sorted_clusters = cluster_performance.sort_values().index

cluster_names = {
    sorted_clusters[0]: "Needs Improvement",
    sorted_clusters[1]: "Average Performers",
    sorted_clusters[2]: "High Performers"
}

df["Performance_Group"] = df["Cluster"].map(cluster_names)


print("\nFinal Employee Performance Groups:")

result = df[
    [
        "Employee",
        "Experience",
        "Salary",
        "Performance_Score",
        "Working_Hours",
        "Performance_Group"
    ]
]

print(result.to_string(index=False))


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Experience",
    y="Performance_Score",
    hue="Performance_Group",
    style="Performance_Group",
    s=120
)

plt.title("Employee Performance Groups")
plt.xlabel("Experience (Years)")
plt.ylabel("Performance Score")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Salary",
    y="Performance_Score",
    hue="Performance_Group",
    s=120
)

plt.title("Salary vs Performance")
plt.xlabel("Salary")
plt.ylabel("Performance Score")

plt.tight_layout()
plt.show()

df.to_csv("employee_performance_groups.csv", index=False)

print("\nProject completed successfully!")
print("Result saved as employee_performance_groups.csv")