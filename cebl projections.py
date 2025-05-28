import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pickle
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 0. Helper Function: Convert Year String to Numeric
# -----------------------------
def convert_year_to_numeric(year_str):
    """
    Convert a school year string "YYYY-YY" to a numeric value by taking the starting year + 0.5.
    For example, "2021-22" becomes 2021.5.
    """
    try:
        start_year = int(year_str.split("-")[0])
        return start_year + 0.5
    except Exception as e:
        print(f"Error converting year: {year_str} -> {e}")
        return np.nan

# -----------------------------
# 1. Data Loading, Preprocessing, and Feature Selection
# -----------------------------
file_path = r"C:\Users\hasan\Documents\python\realgm_reg_yr.csv"
df = pd.read_csv(file_path)
print("Dataset Preview:")
print(df.head())

# Convert "Year" column if it exists
if "Year" in df.columns:
    df["Year_numeric"] = df["Year"].apply(convert_year_to_numeric)
    df.drop(columns=["Year"], inplace=True)
    df.rename(columns={"Year_numeric": "Year"}, inplace=True)

# Define target columns (three outputs)
target_cols = ['CEBL Rank', 'CEBL PER Rank', 'Salary with Bonus']
if not set(target_cols).issubset(df.columns):
    raise ValueError(f"Target columns {target_cols} not found in the CSV file.")

# Exclude specific features from the predictors
excluded_features = ['CEBL FG%', 'CEBL 3P%', 'CEBL PPG', 'CEBL RPG', 'CEBL APG', 'CEBL PER']
X = df.drop(columns=target_cols)
X = X.drop(columns=[col for col in excluded_features if col in X.columns])
# Retain only numeric columns
X = X.select_dtypes(include=[np.number])
y = df[target_cols]

# --- Feature Selection: Drop features with low average importance ---
scaler_fs = StandardScaler()
X_scaled_fs = scaler_fs.fit_transform(X)
rf_multi = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
rf_multi.fit(X_scaled_fs, y)
importances = []
for estimator in rf_multi.estimators_:
    importances.append(estimator.feature_importances_)
avg_importance = np.mean(importances, axis=0)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': avg_importance
}).sort_values(by='Importance', ascending=False)
print("\nComputed Feature Importances:")
print(importance_df)

threshold = importance_df['Importance'].mean()
print(f"\nDropping features with average importance below: {threshold:.4f}")
features_to_drop = importance_df[importance_df['Importance'] < threshold]['Feature'].tolist()
print("Features to drop:", features_to_drop)
X = X.drop(columns=features_to_drop)
print("\nFeatures used after dropping low-importance ones:")
print(X.columns.tolist())

# -----------------------------
# 1a. Scaling and Polynomial Feature Expansion
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
feature_names = poly.get_feature_names_out(X.columns)
print("\nFeatures used for training (after scaling & polynomial expansion):")
print(feature_names)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# -----------------------------
# 2. Model Training, Evaluation, and Per-Target Metrics
# -----------------------------
print("\nOriginal numeric features used (before poly expansion):")
print(X.columns.tolist())

def evaluate_model_per_target(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2_list = []
    mae_list = []
    for i, col in enumerate(y_test.columns):
        r2 = r2_score(y_test.iloc[:, i], preds[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
        r2_list.append(r2)
        mae_list.append(mae)
    return r2_list, mae_list

models = {
    'Linear Regression': MultiOutputRegressor(LinearRegression()),
    'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42)),
    'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, random_state=42)),
    'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(n_estimators=200, random_state=42, objective='reg:squarederror'))
}

results = {}
print("\nPer-Target Evaluation Results (R² and MAE):")
for name, model in models.items():
    r2_list, mae_list = evaluate_model_per_target(model, X_train, X_test, y_train, y_test)
    results[name] = {"R2": r2_list, "MAE": mae_list}
    print(f"\n{name}:")
    for i, target in enumerate(target_cols):
        print(f"  {target}: R² = {r2_list[i]:.3f}, MAE = {mae_list[i]:.2f}")

# -----------------------------
# 4. Neural Network Model using Keras
# -----------------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_nn_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(25, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='he_uniform'))  # Linear activation
    model.compile(loss='mae', optimizer=Adam(learning_rate=0.01))
    return model

nn_model = build_nn_model(input_dim=X_train.shape[1], output_dim=y.shape[1])
nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
nn_preds = nn_model.predict(X_test)
nn_r2_list = []
nn_mae_list = []
for i, col in enumerate(y_test.columns):
    r2 = r2_score(y_test.iloc[:, i], nn_preds[:, i])
    mae = mean_absolute_error(y_test.iloc[:, i], nn_preds[:, i])
    nn_r2_list.append(r2)
    nn_mae_list.append(mae)
print("\nNeural Network Model Evaluation:")
for i, target in enumerate(target_cols):
    print(f"  {target}: R² = {nn_r2_list[i]:.3f}, MAE = {nn_mae_list[i]:.2f}")