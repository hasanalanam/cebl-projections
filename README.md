# CEBL Player Performance & Salary Prediction

A Python-based, multi-output regression pipeline for forecasting key basketball metrics—player rank, PER rank, and salary with bonus—using historical CEBL data. Designed to showcase advanced feature engineering, model selection, and evaluation practices ideal for team analytics and decision support.

## 🚀 Key Features
- **Multi-Target Regression**  
  - Leverages `MultiOutputRegressor` to predict three targets simultaneously.
- **Feature Engineering & Selection**  
  - Year-string conversion helper  
  - Automated importance-based feature filtering via Random Forest  
  - Scaling (`StandardScaler`) and polynomial feature expansion (`PolynomialFeatures`)
- **Model Zoo**  
  - Linear Regression  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  
  - XGBoost Regressor
- **Rigorous Evaluation**  
  - Per-target R² and MAE metrics for fine-grained performance insight

## 🛠️ Tech Stack
- Python 3.x  
- `pandas`, `numpy`  
- `scikit-learn`, `xgboost`  
- Optional: `pickle` for model serialization
