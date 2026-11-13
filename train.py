import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load Dataset
df = pd.read_csv("hotel_bookings.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# 2. EDA - Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Target Variable Distribution
print("\nCancellation Distribution:")
print(df["is_canceled"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x="is_canceled", data=df)
plt.title("Hotel Booking Cancellation Distribution")
plt.savefig("cancellation_distribution.png")
plt.show()

# 4. Remove Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# 5. Drop Data Leakage Columns
df = df.drop(
    ["reservation_status", "reservation_status_date"],
    axis=1
)


# Outlier Detection using IQR Method

numeric_cols = df.select_dtypes(
    include="number"
).drop(columns=["is_canceled"]).columns
outlier_summary = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]
    outlier_summary.append({
        "Column": col,
        "Lower Limit": lower_limit,
        "Upper Limit": upper_limit,
        "Outlier Count": len(outliers)
    })
outlier_df = pd.DataFrame(outlier_summary)
print("Outlier Detection Summary:")
display(outlier_df.sort_values(
    by="Outlier Count",
    ascending=False
))

# 6. Split Features and Target
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Identify Numerical and Categorical Columns
num_cols = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

cat_cols = X_train.select_dtypes(
    include=["object"]
).columns

# 9. Numerical Preprocessing
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# 10. Categorical Preprocessing
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 11. Combine Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 12. Create Random Forest Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

# 13. Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# 14. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 15. Model Evaluation
print("\nModel Evaluation Results:")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 16. Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("Random Forest Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# 17. Save Model
joblib.dump(model, "hotel_bookings_pipeline.pkl")
print("\nModel Saved Successfully!")
