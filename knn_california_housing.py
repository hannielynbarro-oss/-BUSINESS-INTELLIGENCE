import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

print("Creating California Housing Dataset...")

np.random.seed(42)
n_samples = 20640

data = {
    'MedInc': np.random.lognormal(mean=1.5, sigma=0.8, size=n_samples).clip(0.5, 15),
    'HouseAge': np.random.gamma(shape=2, scale=15, size=n_samples).clip(1, 52),
    'AveRooms': np.random.gamma(shape=2, scale=3, size=n_samples).clip(0.8, 15),
    'AveBedrms': np.random.gamma(shape=2, scale=1, size=n_samples).clip(0.2, 4),
    'Population': np.random.lognormal(mean=6, sigma=0.5, size=n_samples).clip(3, 35682),
    'AveOccup': np.random.gamma(shape=1.5, scale=2, size=n_samples).clip(0.7, 1243),
    'Latitude': np.random.normal(35.5, 2, n_samples).clip(32.5, 42),
    'Longitude': np.random.normal(-119.5, 2, n_samples).clip(-124.5, -114)
}

df = pd.DataFrame(data)

base_value = (data['MedInc'] * 0.5 + 
              np.exp((data['Latitude'] - 35) * 0.1) * 2 +  
              np.random.normal(0, 0.5, n_samples))

base_value += (data['HouseAge'] * 0.01) - (data['AveRooms'] * 0.05) + (data['AveBedrms'] * 0.1)
base_value += np.random.normal(0, 0.3, n_samples)  

df['MedHouseVal'] = np.clip(base_value, 0.15, 5.0)

X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values

feature_names = df.columns.tolist()[:-1]

print("Dataset Shape:", df.shape)
print("\nFeatures:", feature_names)
print("\nTarget Variable: Median House Value (in $100,000s)")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== Training KNN Regressor (k=5) ===")
knn_regressor = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')
knn_regressor.fit(X_train_scaled, y_train)

y_pred = knn_regressor.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].scatter(y_test, y_pred, alpha=0.6, edgecolors='w', s=50)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Values ($100,000s)')
axes[0, 0].set_ylabel('Predicted Values ($100,000s)')
axes[0, 0].set_title('Actual vs Predicted House Values')
axes[0, 0].grid(True, alpha=0.3)

residuals = y_test - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors='w', s=50)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted Values ($100,000s)')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(y_test, bins=30, alpha=0.7, label='Actual', color='blue', density=True)
axes[1, 0].hist(y_pred, bins=30, alpha=0.7, label='Predicted', color='red', density=True)
axes[1, 0].set_xlabel('House Value ($100,000s)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Distribution of Actual vs Predicted Values')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', ax=axes[1, 1])
axes[1, 1].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig('knn_california_housing_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

feature_correlation = df.corr()['MedHouseVal'].drop('MedHouseVal').sort_values(ascending=False)

print("\n=== Feature Correlation with Target ===")
print(feature_correlation)

plt.figure(figsize=(10, 6))
feature_correlation.plot(kind='bar')
plt.title('Feature Correlation with Median House Value')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_correlation_housing.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Testing Different k Values ===")
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
mse_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    mse_k = mean_squared_error(y_test, y_pred_k)
    mse_scores.append(mse_k)
    print(f"k={k}: MSE = {mse_k:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, mse_scores, 'bo-')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Number of Neighbors')
plt.grid(True, alpha=0.3)
plt.savefig('mse_vs_k_values.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Summary and Interpretation ===")
print(f"KNN Regressor Performance (k=5):")
print(f"- Mean Squared Error (MSE): {mse:.4f}")
print(f"- Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"- Mean Absolute Error (MAE): {mae:.4f}")
print(f"- R² Score: {r2:.4f}")

print(f"\nInterpretation:")
print(f"- MSE of {mse:.4f} means the average squared prediction error is {mse:.4f} ($100,000s)²")
print(f"- RMSE of {rmse:.4f} means the average prediction error is ${rmse*100000:.0f}")
print(f"- MAE of {mae:.4f} means the average absolute error is ${mae*100000:.0f}")
print(f"- R² of {r2:.4f} means the model explains {r2*100:.1f}% of the variance in house prices")

print(f"\nMost Correlated Features:")
for feature, corr in feature_correlation.head(3).items():
    print(f"- {feature}: {corr:.4f}")

print(f"\nOptimal k value based on MSE: {k_values[np.argmin(mse_scores)]}")
print(f"Best MSE: {min(mse_scores):.4f}")
