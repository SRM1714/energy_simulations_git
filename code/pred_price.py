import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib




def clean_numeric_df(X: pd.DataFrame, y: pd.Series = None):
    """
    Converts all columns to numeric, dropping any that cannot be.
    Removes NaNs. Optionally aligns and returns y.
    """
    X_clean = pd.DataFrame()

    for col in X.columns:
        try:
            # Only keep Series-like columns
            if X[col].ndim == 1:
                numeric_col = pd.to_numeric(X[col], errors='coerce')
                if not numeric_col.isna().all():
                    X_clean[col] = numeric_col
        except Exception as e:
            print(f"Skipping column '{col}' due to error: {e}")
            continue

    X_clean = X_clean.dropna()

    if y is not None:
        y_clean = y.loc[X_clean.index]
        return X_clean, y_clean
    return X_clean



def prepare_price_data(filepath, target_column='Spot_electricidad', test_size=0.2):

    df = pd.read_csv(filepath)
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Drop missing target values
    df = df.dropna(subset=[target_column])
    df = df.fillna(method='ffill')

    # Time-based features
    df['hour'] = df['Fecha'].dt.hour
    df['day_of_week'] = df['Fecha'].dt.dayofweek
    df['month'] = df['Fecha'].dt.month

    # Define features dynamically
    base_features = ['hour', 'day_of_week', 'month']
    dynamic_features = [col for col in df.columns if col not in ['Fecha', target_column] and df[col].dtype in [np.float64, np.int64]]
    valid_features = [f for f in (base_features + dynamic_features) if not df[f].isna().all()]

    X = df[valid_features].fillna(method='ffill').dropna()
    y = df.loc[X.index, target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

    return X_train, X_test, y_train, y_test, df


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    X_train = X_train.apply(pd.to_numeric, errors='coerce').dropna()
    y_train = y_train.loc[X_train.index]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4,
                         objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    return model



def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "Model": name,
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Train MAPE": mape(y_train, y_train_pred),
        "Test MAPE": mape(y_test, y_test_pred),
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),
    }

    return metrics, y_test_pred



def compare_models(results_list):
    return pd.DataFrame(results_list)


def plot_predictions(y_true, y_pred, model_name="Model", samples=200):
    plt.figure(figsize=(12,6))
    plt.plot(y_true.values[:samples], label='Actual', linewidth=2)
    plt.plot(y_pred[:samples], label=f'{model_name} Prediction', linestyle='--')
    plt.title(f"{model_name} - Electricity Price Prediction (First {samples} Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Electricity Price")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    filepath = "/home/guillermo/Documentos/ahpc/energy/data/dataframe.csv"
    X_train, X_test, y_train, y_test, df = prepare_price_data(filepath)

    results = {}

    rf_model = train_random_forest(X_train, y_train)
    rf_metrics, y_rf_pred = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    results['Random Forest'] = rf_metrics

    gb_model = train_gradient_boosting(X_train, y_train)
    gb_metrics, y_gb_pred = evaluate_model(gb_model, X_train, y_train, X_test, y_test, "Gradient Boosting")
    results['Gradient Boosting'] = gb_metrics


    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Drop any remaining NaNs (just in case)
    # Clean X_train
    X_train_clean, y_train_clean = clean_numeric_df(X_train, y_train)
    X_test_clean, y_test_clean = clean_numeric_df(X_test, y_test)

    xgb_model = train_xgboost(X_train_clean, y_train_clean)
    xgb_metrics, y_xgb_pred = evaluate_model(xgb_model, X_train_clean, y_train_clean, X_test_clean, y_test_clean, "XGBoost")


    results['XGBoost'] = xgb_metrics

    # Print results
    results_df = compare_models(list(results.values()))
    print("\n📊 Model Comparison:")
    print(results_df.round(2))

    # Plot one or more models
    plot_predictions(y_test, y_rf_pred, model_name="Random Forest", samples=200)
    plot_predictions(y_test, y_gb_pred, model_name="Gradient Boosting", samples=200)
    plot_predictions(y_test, y_xgb_pred, model_name="XGBoost", samples=200)

    joblib.dump(xgb_model, "model_price.pkl")

    features = X_train.columns.tolist()
    model_bundle = {
        "model": xgb_model,
        "features": features
    }
    joblib.dump(model_bundle, "model_price.pkl")
