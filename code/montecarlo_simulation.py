import numpy as np
import pandas as pd
import joblib
from simulation import simulate_strategies, optimize_strategy_mcmc
import matplotlib.pyplot as plt

def generate_synthetic_weather_day(start_date="2025-07-01"):
    ghi_mu = [0, 100, 300, 600, 700, 600, 400, 200, 50, 0, 0, 0]
    ghi = [max(0, np.random.normal(mu, mu * 0.2 if mu > 0 else 5)) for mu in ghi_mu]
    ghi_full = ghi + ghi[::-1]

    temp_base = np.random.normal(22, 4)
    temp = [temp_base + 5 * np.sin(np.pi * hour / 12) + np.random.normal(0, 1) for hour in range(12)]
    temp_full = temp + temp[::-1]

    df_weather = pd.DataFrame({
        "timestamp": pd.date_range(start=start_date, periods=24, freq="H"),
        "GHI": ghi_full,
        "Temp": temp_full,
    })
    return df_weather


def monte_carlo_simulation(n_simulations=100):
    prod_model = joblib.load("model_production.pkl")
    price_model = joblib.load("model_price.pkl")

    profit_direct_all = []
    profit_battery_all = []
    profit_optimized_all = []

    for i in range(n_simulations):
        weather = generate_synthetic_weather_day(f"2025-07-{(i%30)+1:02d}")
        weather["hour"] = weather["timestamp"].dt.hour
        features = ["hour", "GHI", "Temp"]

        prod_bundle = joblib.load("model_production.pkl")
        prod_model = prod_bundle["model"]
        prod_scaler = prod_bundle["scaler"]
        prod_features = prod_bundle["features"]

        # Create synthetic weather matching the needed features
        # You'll have to approximate MODULE_TEMPERATURE, IS_WEEKEND, DAY_OF_YEAR
        weather["AMBIENT_TEMPERATURE"] = weather["Temp"]
        weather["MODULE_TEMPERATURE"] = weather["Temp"] + np.random.normal(5, 1, size=len(weather))
        weather["IRRADIATION"] = weather["GHI"]
        weather["HOUR"] = weather["timestamp"].dt.hour + weather["timestamp"].dt.minute / 60.0
        weather["DAY_OF_YEAR"] = weather["timestamp"].dt.dayofyear
        weather["IS_WEEKEND"] = (weather["timestamp"].dt.dayofweek >= 5).astype(int)

        # Apply the same scaling
        X_prod_raw = weather[prod_features]
        X_prod_scaled = prod_scaler.transform(X_prod_raw)
        weather["production_kwh"] = prod_model.predict(X_prod_scaled)


        # Predict production
        X_prod = weather[features]
        weather["production_kwh"] = prod_model.predict(X_prod_scaled)

        # Predict price
        price_bundle = joblib.load("model_price.pkl")
        price_model = price_bundle["model"]
        price_features = price_bundle["features"]

        # Create placeholder values for all expected features
        for col in price_features:
            if col not in weather.columns:
                weather[col] = 0  # or use some realistic default if known

        # Rename synthetic features to match price model expectations if applicable
        weather["Radiacion"] = weather["GHI"]
        weather["Temperatura"] = weather["Temp"]

        X_price = weather[price_features]
        weather["predicted_price"] = price_model.predict(X_price)


        df_eval = weather[["timestamp", "production_kwh", "predicted_price"]].copy()
        df_eval = df_eval.set_index("timestamp")

        # Simulate
        profit_direct, profit_battery, battery_soc, _ = simulate_strategies(df_eval)
        actions, profit_mcmc, battery_soc_mcmc, _ = optimize_strategy_mcmc(df_eval)

        profit_direct_all.append(profit_direct[-1])
        profit_battery_all.append(profit_battery[-1])
        profit_optimized_all.append(profit_mcmc[-1])

        print(f"Simulation {i+1}/{n_simulations} complete.")

    return profit_direct_all, profit_battery_all, profit_optimized_all


def plot_montecarlo_results(profit_direct_all, profit_battery_all, profit_optimized_all):
    plt.figure(figsize=(10,6))
    plt.hist(profit_direct_all, bins=20, alpha=0.6, label="Direct sale")
    plt.hist(profit_battery_all, bins=20, alpha=0.6, label="Rule-based battery")
    plt.hist(profit_optimized_all, bins=20, alpha=0.6, label="MCMC optimized")
    plt.xlabel("Cumulative Profit (€)")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo Simulation of Battery Strategy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    profit_direct_all, profit_battery_all, profit_optimized_all = monte_carlo_simulation(n_simulations=2)
    plot_montecarlo_results(profit_direct_all, profit_battery_all, profit_optimized_all)
