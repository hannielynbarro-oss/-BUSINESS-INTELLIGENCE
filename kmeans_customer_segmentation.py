import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

print("Generating Customer Dataset...")
np.random.seed(42)
n_samples = 200

segments = []

segment1 = {
    'income': np.random.normal(25000, 5000, 40),
    'spending': np.random.normal(25, 10, 40)
}

segment2 = {
    'income': np.random.normal(60000, 15000, 40),
    'spending': np.random.normal(50, 15, 40)
}

segment3 = {
    'income': np.random.normal(120000, 20000, 40),
    'spending': np.random.normal(85, 10, 40)
}

segment4 = {
    'income': np.random.normal(100000, 25000, 40),
    'spending': np.random.normal(30, 12, 40)
}

segment5 = {
    'income': np.random.normal(30000, 8000, 40),
    'spending': np.random.normal(70, 15, 40)
}

all_income = np.concatenate([segment1['income'], segment2['income'], segment3['income'], 
                           segment4['income'], segment5['income']])
all_spending = np.concatenate([segment1['spending'], segment2['spending'], segment3['spending'],
                              segment4['spending'], segment5['spending']])

df = pd.DataFrame({
    'Annual_Income': all_income,
    'Spending_Score': all_spending
})

df['Annual_Income'] = df['Annual_Income'].clip(15000, 150000)
df['Spending_Score'] = df['Spending_Score'].clip(1, 100)

print("Dataset Shape:", df.shape)
print("\nDataset Statistics:")
print(df.describe())
print("\nFirst 5 rows:")
print(df.head())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['Annual_Income'], df['Spending_Score'], alpha=0.6, s=50)
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.title('Raw Customer Data')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(df['Annual_Income'], bins=20, alpha=0.7, label='Income', color='blue')
plt.xlabel('Annual Income ($)')
plt.ylabel('Frequency')
plt.title('Income Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('raw_customer_data.png', dpi=300, bbox_inches='tight')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print("\n=== Applying K-Means Clustering (k=5) ===")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = cluster_labels

silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")

cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers_original, 
                                 columns=['Annual_Income', 'Spending_Score'])
cluster_centers_df['Cluster'] = range(5)

print("\nCluster Centers:")
print(cluster_centers_df)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual_Income'], cluster_data['Spending_Score'], 
               c=colors[i], label=f'Cluster {i}', alpha=0.7, s=50)
    
plt.scatter(cluster_centers_df['Annual_Income'], cluster_centers_df['Spending_Score'],
           c='black', marker='X', s=200, linewidths=3, label='Centroids')

plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments with K-Means (k=5)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
cluster_sizes = df['Cluster'].value_counts().sort_index()
plt.bar(cluster_sizes.index, cluster_sizes.values, color=colors)
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Cluster Sizes')
plt.xticks(range(5))
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    plt.hist(cluster_data['Annual_Income'], bins=10, alpha=0.5, 
             label=f'Cluster {i}', color=colors[i])
plt.xlabel('Annual Income ($)')
plt.ylabel('Frequency')
plt.title('Income Distribution by Cluster')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    plt.hist(cluster_data['Spending_Score'], bins=10, alpha=0.5,
             label=f'Cluster {i}', color=colors[i])
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.title('Spending Distribution by Cluster')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
plt.scatter(cluster_centers_df['Annual_Income'], cluster_centers_df['Spending_Score'],
           c=colors, s=300, alpha=0.7)
for i, row in cluster_centers_df.iterrows():
    plt.annotate(f'C{i}', (row['Annual_Income'], row['Spending_Score']), 
                xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
plt.xlabel('Average Annual Income ($)')
plt.ylabel('Average Spending Score')
plt.title('Cluster Centroids')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(X_scaled)
    inertias.append(kmeans_test.inertia_)

plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.axvline(x=5, color='red', linestyle='--', label='k=5')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_customer_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Customer Segment Analysis ===")

segment_analysis = []
for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    avg_income = cluster_data['Annual_Income'].mean()
    avg_spending = cluster_data['Spending_Score'].mean()
    size = len(cluster_data)
    
    if avg_income < 40000 and avg_spending < 40:
        segment_type = "Budget-Conscious"
        description = "Low income, conservative spending"
    elif avg_income < 40000 and avg_spending > 60:
        segment_type = "Impulsive Spenders"
        description = "Low income, high spending (potential debt risk)"
    elif 40000 <= avg_income < 80000 and 40 <= avg_spending <= 60:
        segment_type = "Average Customers"
        description = "Moderate income, balanced spending"
    elif avg_income >= 80000 and avg_spending >= 70:
        segment_type = "Premium Customers"
        description = "High income, high spending (ideal target)"
    elif avg_income >= 80000 and avg_spending < 40:
        segment_type = "Conservative Wealthy"
        description = "High income, low spending (savers)"
    else:
        segment_type = "Mixed Segment"
        description = "Varied characteristics"
    
    segment_analysis.append({
        'Cluster': i,
        'Segment_Type': segment_type,
        'Description': description,
        'Avg_Income': avg_income,
        'Avg_Spending': avg_spending,
        'Size': size,
        'Percentage': (size / len(df)) * 100
    })

segment_df = pd.DataFrame(segment_analysis)
print("\nCustomer Segment Summary:")
print(segment_df[['Cluster', 'Segment_Type', 'Description', 'Avg_Income', 
                 'Avg_Spending', 'Size', 'Percentage']].round(2))

print("\n=== Marketing Recommendations ===")
for _, segment in segment_df.iterrows():
    print(f"\n{segment['Segment_Type']} (Cluster {segment['Cluster']}):")
    print(f"  - {segment['Description']}")
    print(f"  - Size: {segment['Size']} customers ({segment['Percentage']:.1f}%)")
    
    if segment['Segment_Type'] == "Premium Customers":
        print("  - Strategy: VIP programs, exclusive offers, premium products")
    elif segment['Segment_Type'] == "Budget-Conscious":
        print("  - Strategy: Discounts, value packs, budget-friendly options")
    elif segment['Segment_Type'] == "Impulsive Spenders":
        print("  - Strategy: Credit options, payment plans, limited-time offers")
    elif segment['Segment_Type'] == "Conservative Wealthy":
        print("  - Strategy: Investment products, long-term value, quality focus")
    elif segment['Segment_Type'] == "Average Customers":
        print("  - Strategy: Loyalty programs, cross-selling, personalized recommendations")

print(f"\nClustering Performance:")
print(f"- Silhouette Score: {silhouette_avg:.3f} (0.5-1.0 is good)")
print(f"- Total Customers: {len(df)}")
print(f"- Number of Segments: 5")
print(f"- Average segment size: {len(df)/5:.0f} customers")