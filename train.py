
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from scipy.stats import randint, uniform

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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
sns.countplot(data=df, x="is_canceled")
plt.title("Hotel Booking Cancellation Distribution")
plt.xlabel("Is Canceled (0 = No, 1 = Yes)")
plt.ylabel("Number of Bookings")
plt.savefig("cancellation_distribution.png")
plt.show()


# 4. Drop Data Leakage Columns
df = df.drop(columns=[
    "reservation_status",
    "reservation_status_date"
])

print("\nUpdated Shape:", df.shape)


# 5. Handle Missing Values
df["children"] = df["children"].fillna(
    df["children"].median()
)

df["country"] = df["country"].fillna(
    df["country"].mode()[0]
)

df["agent"] = df["agent"].fillna(0)

df["company"] = df["company"].fillna(0)

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())


# 6. Remove Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)
print("Remaining Duplicates:", df.duplicated().sum())


# 7. Outlier Detection using IQR Method
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

print("\nOutlier Detection Summary:")
print(outlier_df.sort_values(
    by="Outlier Count",
    ascending=False
))


# 8. Split Features and Target
X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)


# 9. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# 10. Identify Numerical and Categorical Columns
numeric_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude="number"
).columns.tolist()


# 11. Numerical Preprocessing
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# 12. Categorical Preprocessing
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 13. Combine Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 14. Create XGBoost Pipeline
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),

    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ))
])


# 15. Hyperparameter Tuning
param_dist = {
    "model__n_estimators": randint(100, 500),
    "model__max_depth": randint(3, 9),
    "model__learning_rate": uniform(0.01, 0.19),
    "model__subsample": uniform(0.7, 0.3),
    "model__colsample_bytree": uniform(0.7, 0.3),
    "model__min_child_weight": randint(1, 8),
    "model__gamma": uniform(0, 0.5)
}

random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=param_dist,
    n_iter=10,
    scoring="f1",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


# 16. Train and Tune XGBoost
print("\nTraining XGBoost Model...")
print("Fitting 3 folds for each of 10 candidates")

random_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross Validation F1 Score:")
print(random_search.best_score_)


# 17. Best XGBoost Model
best_xgb = random_search.best_estimator_


# 18. Predictions
y_pred = best_xgb.predict(X_test)
y_prob = best_xgb.predict_proba(X_test)[:, 1]


# 19. Model Evaluation
print("\nTuned XGBoost Results")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 20. Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("XGBoost Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()


# 21. Save Optimized XGBoost Model
joblib.dump(best_xgb, "hotel_booking_xgboost.pkl")

model_size = os.path.getsize(
    "hotel_booking_xgboost.pkl"
) / (1024 * 1024)

print("\nXGBoost Model Saved Successfully!")
print(f"Model Size: {model_size:.2f} MB")
