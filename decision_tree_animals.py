import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

data = {
    'TOOTHED': [True, True, True, True, True, True, True, False, False, False, False, True, True, True, True],
    'HAIR': [True, True, False, True, True, True, False, False, False, False, False, True, True, True, True],
    'BREATHES': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
    'LEGS': [False, False, False, False, False, False, False, True, True, True, True, True, True, True, True],
    'SPECIES': ['mammal', 'mammal', 'reptile', 'mammal', 'mammal', 'mammal', 'reptile', 'bird', 'bird', 'bird', 'bird', 'mammal', 'mammal', 'mammal', 'mammal']
}

df = pd.DataFrame(data)
print("Animal Dataset:")
print(df)
print("\nDataset Shape:", df.shape)

X = df.drop('SPECIES', axis=1)
y = df['SPECIES']

X = X.astype(int)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nFeatures (X):")
print(X.head())
print("\nTarget (y) encoded:")
print(list(zip(y, y_encoded))[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

print("\nClassification Report:")
unique_classes = np.unique(np.concatenate([y_test, y_pred]))
class_names = le.inverse_transform(unique_classes)
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=class_names))

plt.figure(figsize=(20, 12))
plot_tree(dt_classifier, 
          feature_names=X.columns, 
          class_names=le.classes_,
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title('Decision Tree for Animal Classification', fontsize=16)
plt.savefig('decision_tree_animals.png', dpi=300, bbox_inches='tight')
plt.show()

test_case = pd.DataFrame({
    'TOOTHED': [True],
    'HAIR': [False], 
    'BREATHES': [True],
    'LEGS': [False]
})
test_case = test_case.astype(int)

prediction = dt_classifier.predict(test_case)
predicted_species = le.inverse_transform(prediction)[0]

print(f"\nTest Case Prediction:")
print(f"Input: TOOTHED=True, HAIR=False, BREATHES=True, LEGS=False")
print(f"Predicted Species: {predicted_species}")

print(f"\nDecision Path Analysis:")
print("1. Root Node: Check TOOTHED feature")
print("   - Since TOOTHED=True, follow the right branch")
print("2. Next Node: Check HAIR feature") 
print("   - Since HAIR=False, follow the left branch")
print("3. Next Node: Check LEGS feature")
print("   - Since LEGS=False, follow the left branch")
print("4. Final Node: Classification reached")
print(f"   - Predicted class: {predicted_species}")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
