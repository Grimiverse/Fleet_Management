import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
my_data = pd.read_csv(r"C:/Users/grima/OneDrive - Teesside University/DISSERTATION FOLDER/logistics_dataset_with_maintenance_required.csv")

# Label Encoding
from sklearn.preprocessing import LabelEncoder
categorical_cols = ["Make_and_Model", "Vehicle_Type", "Maintenance_Type", "Route_Info", "Weather_Conditions", "Road_Conditions", "Last_Maintenance_Date", "Brake_Condition"]
label_encoder = LabelEncoder()
for col in categorical_cols:
    my_data[col] = label_encoder.fit_transform(my_data[col])

# Log transform
my_data["Load_Capacity_Log"] = np.log1p(my_data["Load_Capacity"])
my_data["Actual_Load_Log"] = np.log1p(my_data["Actual_Load"])
my_data["Usage_Hours_Log"] = np.log1p(my_data["Usage_Hours"])
my_data["Maintenance_Cost_Log"] = np.log1p(my_data["Maintenance_Cost"])
my_data["Battery_Status_Log"] = np.log1p(my_data["Battery_Status"])
my_data["Vibration_Levels_Log"] = np.log1p(my_data["Vibration_Levels"])
my_data["Weather_Conditions_Log"] = np.log1p(my_data["Weather_Conditions"])
my_data["Downtime_Maintenance_Log"] = np.log1p(my_data["Downtime_Maintenance"])

# Define target and drop unnecessary features
Y = my_data['Maintenance_Required']
X_full = my_data.drop(columns=['Vehicle_ID', 'Make_and_Model', 'Year_of_Manufacture', 'Vehicle_Type', 
                               'Route_Info', 'Last_Maintenance_Date', 'Engine_Temperature', 
                               'Predictive_Score', 'Road_Conditions', 
                               'Weather_Conditions', 'Delivery_Times', 'Maintenance_Required', 
                               'Downtime_Maintenance_Log', 'Failure_History', 'Anomalies_Detected'])

# Split full data for feature importance evaluation
X_train_full, X_test_full, Y_train, Y_test = train_test_split(X_full, Y, test_size=0.3, random_state=42, stratify=Y)

# Scale full features
scaler_full = MinMaxScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_full_scaled = scaler_full.transform(X_test_full)

# Train initial model
rf_model_full = RandomForestClassifier(
    max_depth=20,
    min_samples_split=10,
    n_estimators=100,
    random_state=42
)
rf_model_full.fit(X_train_full_scaled, Y_train)

# Evaluate feature importances
feature_importances = rf_model_full.feature_importances_
feature_names = X_full.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
top_features = importance_df.sort_values(by='Importance', ascending=False).head(7)['Feature'].tolist()

print("✅ Top 7 selected features based on importance:\n", top_features)

# Redefine X using top features only
X_selected = my_data[top_features]

# Split again for final training
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.3, random_state=42, stratify=Y)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain on selected features
rf_tuned_model = RandomForestClassifier(
    max_depth=20,
    min_samples_split=10,
    n_estimators=100,
    random_state=42
)
rf_tuned_model.fit(X_train_scaled, Y_train)

# Evaluate
Y_test_pred_tuned = rf_tuned_model.predict(X_test_scaled)

print("\n✅ Model Performance with Top 7 Features:")
print("Accuracy:", accuracy_score(Y_test, Y_test_pred_tuned))
print("\nClassification Report:")
print(classification_report(Y_test, Y_test_pred_tuned))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_test_pred_tuned))

# Save model and scaler
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(rf_tuned_model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save selected feature list for FastAPI input
with open("selected_features.pkl", "wb") as f:
    pickle.dump(top_features, f)
    
with open("X_test_scaled.pkl", "wb") as f:
    pickle.dump(X_test_scaled, f)

with open("Y_test.pkl", "wb") as f:
    pickle.dump(Y_test, f)
print("\n✅ Model, scaler, and selected features saved successfully!")
