import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

print("Creating Bank Marketing Dataset...")

np.random.seed(42)
n_samples = 10000

data = {
    'age': np.random.normal(40, 12, n_samples).astype(int).clip(18, 95),
    'job': np.random.choice(['admin.', 'technician', 'services', 'management', 'retired', 
                           'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 
                           'self-employed', 'student', 'unknown'], n_samples, p=[0.25, 0.15, 0.10, 0.15, 0.08, 0.12, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]),
    'marital': np.random.choice(['married', 'single', 'divorced'], n_samples, p=[0.6, 0.3, 0.1]),
    'education': np.random.choice(['university.degree', 'high.school', 'basic.9y', 'basic.6y', 
                                 'basic.4y', 'professional.course', 'unknown', 'illiterate'], 
                                n_samples, p=[0.3, 0.25, 0.15, 0.1, 0.08, 0.07, 0.04, 0.01]),
    'default': np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.8, 0.05, 0.15]),
    'housing': np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.45, 0.5, 0.05]),
    'loan': np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.85, 0.1, 0.05]),
    'contact': np.random.choice(['cellular', 'telephone'], n_samples, p=[0.7, 0.3]),
    'month': np.random.choice(['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 
                              n_samples, p=[0.08, 0.07, 0.15, 0.12, 0.14, 0.13, 0.08, 0.07, 0.10, 0.06]),
    'day_of_week': np.random.choice(['mon', 'tue', 'wed', 'thu', 'fri'], n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
    'duration': np.random.exponential(180, n_samples).astype(int).clip(0, 3000),
    'campaign': np.random.poisson(2, n_samples).clip(1, 50),
    'pdays': np.random.exponential(200, n_samples).astype(int),
    'previous': np.random.poisson(0.5, n_samples).clip(0, 10),
    'poutcome': np.random.choice(['nonexistent', 'failure', 'success'], n_samples, p=[0.85, 0.1, 0.05]),
    'emp.var.rate': np.random.normal(0.05, 1.5, n_samples),
    'cons.price.idx': np.random.normal(93.5, 0.8, n_samples),
    'cons.conf.idx': np.random.normal(-40, 5, n_samples),
    'euribor3m': np.random.normal(2.5, 2.0, n_samples),
    'nr.employed': np.random.normal(5100, 100, n_samples)
}

df = pd.DataFrame(data)

success_prob = 0.11
success_prob += (df['duration'] > 300) * 0.15
success_prob += (df['poutcome'] == 'success') * 0.25
success_prob += (df['education'] == 'university.degree') * 0.05
success_prob += (df['age'].between(25, 60)) * 0.03
success_prob = np.clip(success_prob, 0.05, 0.8)

df['y'] = np.random.random(n_samples) < success_prob
df['y'] = df['y'].map({True: 'yes', False: 'no'})

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nTarget Variable Distribution:")
print(df['y'].value_counts())
print("\nTarget Variable Percentage:")
print(df['y'].value_counts(normalize=True) * 100)

print("\nMissing Values:")
print(df.isnull().sum())

print("\n=== Data Preprocessing ===")

df['y'] = df['y'].map({'yes': 1, 'no': 0})

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('y')

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop('y', axis=1)
y = df['y']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

print("\n=== Training Random Forest Classifier ===")
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)

rf_classifier.fit(X_train_scaled, y_train)

y_pred = rf_classifier.predict(X_test_scaled)
y_pred_proba = rf_classifier.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

if accuracy >= 0.80:
    print("✅ Target accuracy (≥80%) achieved!")
else:
    print("❌ Target accuracy (≥80%) not achieved")

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Bank Marketing Dataset')
plt.savefig('confusion_matrix_bank.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Top 10 Most Important Features ===")
print(feature_importance.head(10))

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(data=top_features, x='Importance', y='Feature')
plt.title('Top 15 Feature Importance - Bank Marketing Dataset')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_bank.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Summary of Findings ===")
print(f"Model Performance:")
print(f"- Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"- Training samples: {X_train_scaled.shape[0]}")
print(f"- Test samples: {X_test_scaled.shape[0]}")
print(f"- Features used: {X_train_scaled.shape[1]}")

print(f"\nMost Important Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f"- {row['Feature']}: {row['Importance']:.4f}")

print(f"\nConfusion Matrix Interpretation:")
tn, fp, fn, tp = cm.ravel()
print(f"- True Negatives (No correctly predicted): {tn}")
print(f"- False Positives (No incorrectly predicted as Yes): {fp}")
print(f"- False Negatives (Yes incorrectly predicted as No): {fn}")
print(f"- True Positives (Yes correctly predicted): {tp}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\nKey Metrics:")
print(f"- Precision for 'Yes': {precision:.4f}")
print(f"- Recall for 'Yes': {recall:.4f}")
