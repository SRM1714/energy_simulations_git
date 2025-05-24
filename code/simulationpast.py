import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_prepare_data(price_path, solar_path):
    df = pd.read_csv(price_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.set_index("timestamp").asfreq("H")

    solar = pd.read_csv(solar_path, parse_dates=["timestamp"])
    solar = solar.set_index("timestamp").asfreq("H")

    df = df.join(solar, how="inner")
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["lag1"] = df["price"].shift(1)
    df["rolling_mean_24h"] = df["price"].rolling(24).mean()
    df = df.dropna()
    return df

def use_real_prices(df):
    eval_size = int(len(df) * 0.1)
    df_eval = df.iloc[-eval_size:].copy()
    df_eval["predicted_price"] = df_eval["price"]  # usamos precio real
    return df_eval


def simulate_strategies(df_eval,
                        battery_capacity=10,
                        max_c_rate=2,
                        efficiency=0.9,
                        battery_cost=5000,
                        battery_life_cycles=5000):

    cost_degradation_per_kwh = battery_cost / (battery_capacity * battery_life_cycles)

    battery = 0
    profit_b = 0
    profit_d = 0
    battery_soc = []
    profit_battery = []
    profit_direct = []

    low_threshold = np.percentile(df_eval["predicted_price"] / 1000, 30)
    high_threshold = np.percentile(df_eval["predicted_price"] / 1000, 70)

    for _, row in df_eval.iterrows():
        price_kwh = row["predicted_price"] / 1000
        production = row["production_kwh"]

        # Estrategia directa: vender toda la producción
        profit_d += production * price_kwh
        profit_direct.append(profit_d)

        # Estrategia con batería
        energy_to_store = 0
        energy_to_sell = 0

        if price_kwh < low_threshold:
            charge_possible = min(max_c_rate, battery_capacity - battery, production)
            energy_to_store = charge_possible * efficiency
            battery += energy_to_store
            energy_to_sell = production - charge_possible
            profit_b += energy_to_sell * price_kwh

        else:
            # Vender desde batería si rentable (ingreso > coste de degradación)
            if price_kwh > high_threshold and battery > 0:
                discharge = min(max_c_rate, battery)
                revenue = discharge * price_kwh
                cost = discharge * cost_degradation_per_kwh
                profit = revenue - cost
                if profit > 0:
                    battery -= discharge
                    profit_b += profit

            # Venta directa de producción actual
            profit_b += production * price_kwh

        battery_soc.append(battery)
        profit_battery.append(profit_b)

    return profit_direct, profit_battery, battery_soc, df_eval.index


def plot_results(timestamps, profit_direct, profit_battery, battery_soc):
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(timestamps, profit_direct, label="Ingresos sin batería (venta directa)  wo battery")
    plt.plot(timestamps, profit_battery, label="Ingresos con batería (optimización)   with battery")
    plt.ylabel("Ingresos acumulados (€)")
    plt.legend()
    plt.title("Comparativa de ingresos acumulados (income from energy sell)")

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, battery_soc, label="Nivel de batería (kWh)", color="tab:blue")
    plt.ylabel("kWh")
    plt.xlabel("Tiempo")
    plt.legend()
    plt.title("Evolución del estado de carga de la batería (battery charge evolution)")

    plt.tight_layout()
    plt.show()


# ---- EJECUCIÓN PRINCIPAL ----
price_file = "/home/guillermo/Documentos/ahpc/energy/data/energy-charts_Electricity_production_and_spot_prices_in_Germany_in_2025.csv"
solar_file = "data/Simulated_Solar_Production.csv"

#df_eval = train_price_model(df)
df = load_and_prepare_data(price_file, solar_file)
df_eval = use_real_prices(df)
profit_direct, profit_battery, battery_soc, timestamps = simulate_strategies(df_eval)
plot_results(timestamps, profit_direct, profit_battery, battery_soc)
