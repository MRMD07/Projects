import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------------
# Load data
# ---------------------
df = pd.read_csv("vwap_rms_backtest_results.csv")

# Convert result column to binary label (1 = win, 0 = loss/no_exit)
df['label'] = df['result_D'].apply(lambda x: 1 if x == 'win' else 0)

# Select ML feature columns
features = ["atr", "vwaps", "rms", "rms_sma", "vwaps_slope", "zscore", "volume"]
X = df[features]
y = df["label"]

# ---------------------
# Train-test split
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ---------------------
# Train ML Model
# ---------------------
model = RandomForestClassifier(n_estimators=300, max_depth=6)
model.fit(X_train, y_train)

# ---------------------
# Evaluate model
# ---------------------
print("\nModel Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred := model.predict(X_test)))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------
# Export Model
# ---------------------
joblib.dump(model, "model_filter.pkl")
print("\nSaved trained model to model_filter.pkl")

# Show feature importance (helps convert to Pine Script later)
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance:\n", importances)
