import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

print("Dataset Shape:", df.shape)
print("\nFeatures:", feature_names)
print("Target Classes:", target_names)
print("\nFirst 5 rows:")
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

y_pred = svm_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = ['red', 'green', 'blue']
for i, species in enumerate(target_names):
    mask = y == i
    plt.scatter(X[mask, 0], X[mask, 2], c=colors[i], label=species, alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Actual Iris Species')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for i, species in enumerate(target_names):
    mask = y_pred == i
    plt.scatter(X_test[mask, 0], X_test[mask, 2], c=colors[i], label=f'Pred: {species}', alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('SVM Predicted Species (Test Set)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_iris_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

feature_importance = np.abs(svm_classifier.coef_).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

species_performance = {}
for i, species in enumerate(target_names):
    species_mask = (y_test == i)
    species_accuracy = accuracy_score(y_test[species_mask], y_pred[species_mask])
    species_performance[species] = species_accuracy

print("\nSpecies-wise Accuracy:")
for species, accuracy in species_performance.items():
    print(f"{species}: {accuracy:.4f}")

best_species = max(species_performance, key=species_performance.get)
print(f"\nBest identified species: {best_species} ({species_performance[best_species]:.4f} accuracy)")
