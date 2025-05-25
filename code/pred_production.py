import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib




def prepare_data(weather_path, generation_path, test_size=0.15):
    # Load data
    weather_df = pd.read_csv(weather_path)
    generation_df = pd.read_csv(generation_path)

    # Parse datetime formats
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    generation_df['DATE_TIME'] = pd.to_datetime(generation_df['DATE_TIME'], format='%d-%m-%Y %H:%M')

    # Merge on timestamp and plant ID
    merged_df = pd.merge(weather_df, generation_df, on=['DATE_TIME', 'PLANT_ID'])

    # Feature engineering
    merged_df['HOUR'] = merged_df['DATE_TIME'].dt.hour + merged_df['DATE_TIME'].dt.minute / 60.0
    merged_df['DAY_OF_YEAR'] = merged_df['DATE_TIME'].dt.dayofyear
    merged_df['IS_WEEKEND'] = (merged_df['DATE_TIME'].dt.dayofweek >= 5).astype(int)

    # Drop NaNs
    merged_df = merged_df.dropna()

    # Features and target
    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'HOUR', 'DAY_OF_YEAR', 'IS_WEEKEND']
    X = merged_df[features]
    y = merged_df['AC_POWER']

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler, merged_df



def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model



def train_knn(X_train, y_train, k=5):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def train_ann(X_train, y_train, input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Optional: Early stopping to avoid overfitting
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train,
              validation_split=0.2,
              epochs=50,
              batch_size=32,
              callbacks=[es],
              verbose=0)  # Set verbose=1 if you want to see progress

    return model


def prepare_lstm_sequences(X_scaled, y, time_steps=4):
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X_scaled)):
        X_seq.append(X_scaled[i - time_steps:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)



def train_lstm(X_train, y_train, input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=input_shape, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=0)
    return model



def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=6):
    model = XGBRegressor(n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         max_depth=max_depth,
                         objective='reg:squarederror',
                         verbosity=0)
    model.fit(X_train, y_train)
    return model




def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model", is_lstm=False):
    y_train_pred = model.predict(X_train).flatten()
    y_test_pred = model.predict(X_test).flatten()

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



def plot_predictions(y_true, y_pred, timestamps, model_name, zoom='daily', start_index=0):
    """
    zoom: 'daily' (24h) or 'weekly' (7 days)
    start_index: where to start in the testing period
    """
    assert zoom in ['daily', 'weekly']

    if zoom == 'daily':
        hours = 24
        title = f"{model_name} Prediction - Daily View"
    else:
        hours = 24 * 7
        title = f"{model_name} Prediction - Weekly View"

    end_index = start_index + hours
    plt.figure(figsize=(14, 5))
    plt.plot(timestamps[start_index:end_index], y_true[start_index:end_index], label='True', marker='o')
    plt.plot(timestamps[start_index:end_index], y_pred[start_index:end_index], label='Predicted', marker='x')
    plt.title(title)
    plt.ylabel("AC Power")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()



def compare_models(results_list):
    df = pd.DataFrame(results_list)
    return df


if __name__ == "__main__":
    #data1
    weather_path = "/home/guillermo/Documentos/ahpc/energy/data/Plant_1_Weather_Sensor_Data.csv"
    generation_path = "/home/guillermo/Documentos/ahpc/energy/data/Plant_1_Generation_Data.csv"

    # data2
    #weather_df = pd.read_csv('/home/guillermo/Documentos/ahpc/energy/data/Plant_2_Weather_Sensor_Data.csv')
    #generation_df = pd.read_csv('/home/guillermo/Documentos/ahpc/energy/data/Plant_2_Generation_Data.csv')

    X_train, X_test, y_train, y_test, scaler, merged_df = prepare_data(weather_path, generation_path)
    results = []

    # --- Random Forest ---
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics, y_rf_pred = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    results.append(rf_metrics)


    # k-NN
    knn_model = train_knn(X_train, y_train, k=5)
    knn_metrics, y_knn_pred = evaluate_model(knn_model, X_train, y_train, X_test, y_test, "k-NN (k=5)")
    results.append(knn_metrics)

    # ANN
    ann_model = train_ann(X_train, y_train, input_dim=X_train.shape[1])
    ann_metrics, y_ann_pred = evaluate_model(ann_model, X_train, y_train, X_test, y_test, "ANN")
    results.append(ann_metrics)

    # --- XGBoost ---
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics, y_xgb_pred = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
    results.append(xgb_metrics)


    # LSTM
    # Rebuild full scaled set
    full_X_scaled = np.vstack((X_train, X_test))
    full_y = np.hstack((y_train, y_test))

    X_lstm_all, y_lstm_all = prepare_lstm_sequences(full_X_scaled, full_y, time_steps=4)

    split_index = int(0.85 * len(X_lstm_all))
    X_train_lstm, X_test_lstm = X_lstm_all[:split_index], X_lstm_all[split_index:]
    y_train_lstm, y_test_lstm = y_lstm_all[:split_index], y_lstm_all[split_index:]

    lstm_model = train_lstm(X_train_lstm, y_train_lstm, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
    lstm_metrics, y_lstm_pred = evaluate_model(lstm_model, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, "LSTM")
    results.append(lstm_metrics)

    predictions = {
        "k-NN": y_knn_pred,
        "ANN": y_ann_pred,
        "LSTM": y_lstm_pred,
        "Random Forest": y_rf_pred,
        "XGBoost": y_xgb_pred
    }

    for name, y_pred in predictions.items():
        plot_predictions(y_true=y_test.values, y_pred=y_pred,
                        timestamps=merged_df['DATE_TIME'].values[-len(y_test):],
                        model_name=name, zoom='daily', start_index=24)


    # Show results
    results_df = compare_models(results)
    print("\nFinal Model Comparison:")
    print(results_df.round(2))

    # Explicitly define features list to save with model
    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'HOUR', 'DAY_OF_YEAR', 'IS_WEEKEND']

    model_bundle = {
        "model": xgb_model,
        "scaler": scaler,
        "features": features  # this is already defined earlier
    }
    joblib.dump(model_bundle, "model_production.pkl")

