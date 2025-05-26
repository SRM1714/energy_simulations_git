import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Cargar modelo previamente entrenado
model_bundle = joblib.load("/home/guillermo/Documentos/ahpc/energy_simulations_git/model_production.pkl")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
features = model_bundle["features"]

def predict_production_from_weather(weather_df):
    X = weather_df[features]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled).flatten()
    return np.clip(y_pred, 0, None)  # Clip negative values if any



def generate_hourly_irradiance_days(days=7, month=6):
    """Generates realistic hourly irradiance for a number of days with seasonal variation and noise."""
    irradiance_profiles = {
        1: [0, 0, 0, 0, 10, 50, 100, 200, 300, 400, 500, 550, 500, 400, 300, 200, 100, 50, 10, 0, 0, 0, 0, 0],
        6: [0, 0, 0, 10, 50, 150, 300, 500, 600, 700, 750, 800, 750, 700, 600, 500, 300, 150, 50, 10, 0, 0, 0, 0],
        12: [0, 0, 0, 0, 20, 100, 200, 300, 400, 450, 500, 550, 500, 450, 400, 300, 200, 100, 50, 10, 0, 0, 0, 0],
    }
    base_profile = np.array(irradiance_profiles.get(month, irradiance_profiles[6])) / 1000  # kWh/m²
    efficiency = 0.2
    area = 50  # m² of panels
    daily_output = base_profile * efficiency * area

    full = []
    for _ in range(days):
        noise = np.random.normal(0, daily_output * 0.15)
        irradiated = np.clip(daily_output + noise, 0, None)
        full.extend(irradiated)

    return np.array(full)


def generate_hourly_weather_days(days=7, month=6):
    total_hours = days * 24
    hours = np.tile(np.arange(24), days)
    dates = pd.date_range(f"2020-{month:02d}-01", periods=total_hours, freq='H')

    irradiance = generate_hourly_irradiance_days(days, month)

    # Simulate temperatures with daily variation
    ambient_temp = 25 + 5 * np.sin(2 * np.pi * (hours - 6) / 24) + np.random.normal(0, 1, total_hours)
    module_temp = ambient_temp + 10 + np.random.normal(0, 1, total_hours)

    df = pd.DataFrame({
        "DATE_TIME": dates,
        "AMBIENT_TEMPERATURE": ambient_temp,
        "MODULE_TEMPERATURE": module_temp,
        "IRRADIATION": irradiance,
        "HOUR": dates.hour + dates.minute / 60,
        "DAY_OF_YEAR": dates.dayofyear,
        "IS_WEEKEND": (dates.dayofweek >= 5).astype(int)
    })
    return df




def simulate_production_with_model_error(irradiance_kwh, model_error_pct=0.05):
    error = np.random.normal(0, irradiance_kwh * model_error_pct)
    return np.clip(irradiance_kwh + error, 0, None)



def generate_residential_demand_days(days=7):
    hourly_mean = np.array([
        0.3, 0.3, 0.25, 0.2, 0.2, 0.4, 0.7, 1.0, 1.2, 1.0, 0.9, 0.8,
        0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 1.6, 1.0, 0.7, 0.5, 0.4
    ])
    hourly_std = hourly_mean * 0.15
    full_demand = []

    for _ in range(days):
        day = np.random.normal(hourly_mean, hourly_std)
        full_demand.extend(np.clip(day, 0, None))

    return np.array(full_demand)



def simulate_self_sufficiency(production, demand, battery_capacity, max_c_rate, efficiency):
    battery = battery_capacity  # Start fully charged
    grid_used = 0

    for prod, cons in zip(production, demand):
        used_directly = min(prod, cons)
        remaining_need = cons - used_directly
        surplus = prod - used_directly

        # Use battery if needed
        if remaining_need > 0:
            discharge = min(remaining_need, battery, max_c_rate)
            battery -= discharge
            remaining_need -= discharge
            grid_used += remaining_need

        # Store surplus in battery
        if surplus > 0:
            charge = min(surplus, battery_capacity - battery, max_c_rate)
            battery += charge * efficiency

        battery = max(0, min(battery, battery_capacity))

    return grid_used == 0, grid_used



def monte_carlo_self_sufficiency(n_simulations=100, days=7, month=6,
                                  battery_capacity=10,
                                  max_c_rate=2,
                                  efficiency=0.9):
    results = []

    for _ in range(n_simulations):
        #### independent distributions
        weather_df = generate_hourly_weather_days(args.days, args.month)
        production = predict_production_from_weather(weather_df) * args.area


        demand = generate_residential_demand_days(days)


        ### acoounting correlations
        # df = generate_correlated_weather_and_demand(args.days, args.month)
        # production = predict_production_from_weather(df) * args.area
        # demand = df["DEMAND"].values


        is_self_sufficient, grid_kwh = simulate_self_sufficiency(
            production, demand,
            battery_capacity, max_c_rate, efficiency
        )

        results.append({
            "self_sufficient": is_self_sufficient,
            "grid_energy_kwh": grid_kwh
        })

    df = pd.DataFrame(results)
    percent = 100 * df["self_sufficient"].mean()
    print(f"\nFully self-sufficient in {percent:.1f}% of the simulations.")
    return df


def generate_correlated_weather_and_demand(days=7, month=6):
    total_hours = days * 24
    hours = np.tile(np.arange(24), days)
    dates = pd.date_range(f"2020-{month:02d}-01", periods=total_hours, freq='H')

    # --- Base means for each variable ---
    irradiance_mean = np.array([
        0, 0, 0, 10, 50, 150, 300, 500, 600, 700, 750, 800,
        750, 700, 600, 500, 300, 150, 50, 10, 0, 0, 0, 0
    ])[:24] / 1000  # kWh/m²
    ambient_temp_mean = 25 + 5 * np.sin(2 * np.pi * (np.arange(24) - 6) / 24)
    demand_mean = np.array([
        0.3, 0.3, 0.25, 0.2, 0.2, 0.4, 0.7, 1.0, 1.2, 1.0, 0.9, 0.8,
        0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 1.6, 1.0, 0.7, 0.5, 0.4
    ])  # kWh/h

    # Repeat for each day
    irradiance = []
    ambient_temp = []
    demand = []

    # --- Correlation matrix: [irradiance, temperature, demand] ---
    corr_matrix = np.array([
        [1.0,  0.6, -0.7],
        [0.6,  1.0, -0.5],
        [-0.7, -0.5, 1.0]
    ])

    std_devs = [0.1, 2.0, 0.3]  # Rough deviations for each variable

    cov_matrix = np.diag(std_devs) @ corr_matrix @ np.diag(std_devs)

    for h in range(total_hours):
        hour = h % 24
        means = [irradiance_mean[hour], ambient_temp_mean[hour], demand_mean[hour]]
        sample = np.random.multivariate_normal(means, cov_matrix)

        irradiance.append(max(sample[0], 0))
        ambient_temp.append(sample[1])
        demand.append(max(sample[2], 0))

    df = pd.DataFrame({
        "DATE_TIME": dates,
        "IRRADIATION": irradiance,
        "AMBIENT_TEMPERATURE": ambient_temp,
        "MODULE_TEMPERATURE": np.array(ambient_temp) + 10 + np.random.normal(0, 1, total_hours),
        "DEMAND": demand,
        "HOUR": dates.hour + dates.minute / 60,
        "DAY_OF_YEAR": dates.dayofyear,
        "IS_WEEKEND": (dates.dayofweek >= 5).astype(int)
    })

    return df



def plot_results(df):
    plt.figure(figsize=(10,6))
    plt.hist(df["grid_energy_kwh"], bins=20, edgecolor="black", color="lightgreen")
    plt.xlabel("Grid Energy Used (kWh)")
    plt.ylabel("Number of Simulations")
    plt.title("Grid Usage Distribution in Self-Sufficiency Simulations")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Energy self-sufficiency simulation")
    parser.add_argument("--month", type=int, default=6, help="Month to simulate (1-12)")
    parser.add_argument("--battery", type=float, default=10, help="Battery capacity (kWh)")
    parser.add_argument("--c_rate", type=float, default=2, help="Max charge/discharge rate (kW)")
    parser.add_argument("--efficiency", type=float, default=0.9, help="Battery round-trip efficiency")
    parser.add_argument("--simulations", type=int, default=100, help="Number of Monte Carlo simulations")
    parser.add_argument("--days", type=int, default=7, help="Number of days to simulate")
    parser.add_argument("--area", type=float, default=50.0, help="Panel area in square meters")

    args = parser.parse_args()

    df_results = monte_carlo_self_sufficiency(
        n_simulations=args.simulations,
        days=args.days,
        month=args.month,
        battery_capacity=args.battery,
        max_c_rate=args.c_rate,
        efficiency=args.efficiency
    )

    plot_results(df_results)




#python3 self_sufficient_simulation.py --days 30 --month 7 --battery 15 --c_rate 3 --efficiency 0.9 --simulations 500 --area 50
