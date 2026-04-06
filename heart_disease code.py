import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib, json, os
from datetime import datetime

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv(r"C:\Users\MUNIKA\Desktop\SEM 8\CSE 435\cleaned_merged_heart_dataset.csv")

print("Shape:", df.shape)
print(df.isnull().sum())

# 🔴 Change this if target column name is different
target_column = "target"

# ==============================
# 2. Split Features & Target
# ==============================

X = df.drop(target_column, axis=1)
y = df[target_column]

# ==============================
# 3. Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4. Scaling (For LR & KNN)
# ==============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 5. Train 3 Models
# ==============================

def evaluate(model, X_test, y_test):
    return accuracy_score(y_test, model.predict(X_test))

# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
lr_acc = evaluate(lr, X_test_scaled, y_test)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)
rf_acc = evaluate(rf, X_test, y_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
knn_acc = evaluate(knn, X_test_scaled, y_test)

# ==============================
# 6. Compare Models
# ==============================

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "KNN"],
    "Accuracy": [lr_acc, rf_acc, knn_acc]
})

print("\nModel Comparison:")
print(comparison)

best_model_name = comparison.loc[comparison["Accuracy"].idxmax(), "Model"]
print("\nBest Model:", best_model_name)

# ==============================
# 7. Save & Assign Best Model
# ==============================

os.makedirs("models", exist_ok=True)

if best_model_name == "Logistic Regression":
    best_model = lr
    joblib.dump(lr, "models/heart_best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

elif best_model_name == "Random Forest":
    best_model = rf
    joblib.dump(rf, "models/heart_best_model.pkl")

else:
    best_model = knn
    joblib.dump(knn, "models/heart_best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

# ==============================
# 8. Feature Importance (If RF Wins)
# ==============================

if best_model_name == "Random Forest":
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Important Features:")
    print(importance.head(10))

    plt.figure()
    plt.barh(importance["Feature"], importance["Importance"])
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.show()

# ==============================
# 9. Prediction Function
# ==============================

def predict_heart_risk(input_data):
    input_df = pd.DataFrame([input_data])

    prediction = best_model.predict(input_df)[0]
    probability = best_model.predict_proba(input_df)[0][1]

    if probability < 0.33:
        risk = "Low Risk"
    elif probability < 0.66:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return prediction, probability, risk

# ==============================
# 10. Log Accuracy
# ==============================

log_data = {
    "timestamp": str(datetime.now()),
    "LogisticRegression_Accuracy": lr_acc,
    "RandomForest_Accuracy": rf_acc,
    "KNN_Accuracy": knn_acc,
    "Best_Model": best_model_name
}

with open("model_logs.json", "a") as f:
    f.write(json.dumps(log_data) + "\n")

print("\nTraining Completed Successfully!")