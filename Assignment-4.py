import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Task 1: Data Understanding
# ==========================================
# 1. Load the dataset using Pandas
df = pd.read_csv('data.csv')

# 2. Display the first five records
print("--- First 5 Records ---")
print(df.head())

# 3. Identify Numerical features and Target variable
print("\nTarget Variable: 'diagnosis'")
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numerical Features:", numerical_features)

# 4. Display the dataset information and summary statistics
print("\n--- Dataset Information ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# ==========================================
# Task 2: Data Preprocessing
# ==========================================
# 1. Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 2. Remove unnecessary columns 
df = df.drop(columns=['id', 'Unnamed: 32'])

# 3. Encode the target variable
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis']) 

# Separate features (X) and target (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 4. Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalize or standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Task 3: Model Development
# ==========================================
# 1 & 2. Train a KNN classifier using K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 3. Predict the class labels for the test dataset
y_pred = knn.predict(X_test_scaled)

# ==========================================
# Task 4: Model Evaluation
# ==========================================
print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign (B)', 'Malignant (M)'], 
            yticklabels=['Benign (B)', 'Malignant (M)'])
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix - KNN (K=5)')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")
