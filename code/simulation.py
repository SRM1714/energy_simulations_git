import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load and prepare data
# -------------------------------
def load_and_prepare_data(price_path, solar_path):
    df = pd.read_csv(price_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.set_index("timestamp").asfreq("h")

    solar = pd.read_csv(solar_path, parse_dates=["timestamp"])
    solar = solar.set_index("timestamp").asfreq("h")

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
    df_eval["predicted_price"] = df_eval["price"]
    return df_eval


# -------------------------------
# Rule-based simulation
# -------------------------------
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

        profit_d += production * price_kwh
        profit_direct.append(profit_d)

        if price_kwh < low_threshold:
            charge_possible = min(max_c_rate, battery_capacity - battery, production)
            energy_to_store = charge_possible * efficiency
            battery += energy_to_store
            energy_to_sell = production - charge_possible
            profit_b += energy_to_sell * price_kwh

        else:
            if price_kwh > high_threshold and battery > 0:
                discharge = min(max_c_rate, battery)
                revenue = discharge * price_kwh
                cost = discharge * cost_degradation_per_kwh
                profit = revenue - cost
                if profit > 0:
                    battery -= discharge
                    profit_b += profit

            profit_b += production * price_kwh

        battery_soc.append(battery)
        profit_battery.append(profit_b)

    return profit_direct, profit_battery, battery_soc, df_eval.index


def generate_rule_based_actions(df_eval, battery_capacity=10, max_c_rate=2):
    low_threshold = np.percentile(df_eval["predicted_price"] / 1000, 30)
    high_threshold = np.percentile(df_eval["predicted_price"] / 1000, 70)

    actions = []
    battery = 0

    for _, row in df_eval.iterrows():
        price_kwh = row["predicted_price"] / 1000
        production = row["production_kwh"]

        if price_kwh < low_threshold and battery < battery_capacity:
            action = -min(max_c_rate, production, battery_capacity - battery)  # charge
        elif price_kwh > high_threshold and battery > 0:
            action = min(max_c_rate, battery)  # discharge
        else:
            action = 0

        battery = max(0, min(battery_capacity, battery + action))  # update for next iteration
        actions.append(action)

    return np.array(actions)


# -------------------------------
# Simulate with custom actions
# -------------------------------
def simulate_with_actions(df_eval, actions, battery_capacity, max_c_rate, efficiency, battery_cost, battery_life_cycles):
    cost_degradation_per_kwh = battery_cost / (battery_capacity * battery_life_cycles)

    battery = 0
    profit = 0
    battery_soc = []
    profits = []

    for i, (_, row) in enumerate(df_eval.iterrows()):
        price_kwh = row["predicted_price"] / 1000
        production = row["production_kwh"]
        action = np.clip(actions[i], -max_c_rate, max_c_rate)

        if action < 0:
            charge = min(-action, production, battery_capacity - battery)
            actual_charge = charge * efficiency
            battery += actual_charge
            sold = production - charge
            profit += sold * price_kwh

        elif action > 0:
            discharge = min(action, battery)
            revenue = discharge * price_kwh
            cost = discharge * cost_degradation_per_kwh
            profit += max(0, revenue - cost)
            battery -= discharge
            profit += production * price_kwh

        else:
            profit += production * price_kwh

        battery_soc.append(battery)
        profits.append(profit)

    return profit, profits, battery_soc


# -------------------------------
# MCMC Optimization
# -------------------------------
def optimize_strategy_mcmc(df_eval, battery_capacity=10, max_c_rate=2, efficiency=0.9,
                           battery_cost=5000, battery_life_cycles=5000,
                           iterations=10000, initial_temp=1.0, sigma=1.0, cooling_rate=0.9995):
    n = len(df_eval)
    actions = generate_rule_based_actions(df_eval, battery_capacity, max_c_rate)
    best_profit, _, _ = simulate_with_actions(df_eval, actions, battery_capacity, max_c_rate,
                                              efficiency, battery_cost, battery_life_cycles)

    temperature = initial_temp

    for i in range(iterations):
        idx = np.random.randint(n)
        proposal = actions.copy()
        proposal[idx] += np.random.normal(0, sigma)
        proposal[idx] = np.clip(proposal[idx], -max_c_rate, max_c_rate)

        new_profit, _, _ = simulate_with_actions(df_eval, proposal, battery_capacity, max_c_rate,
                                                 efficiency, battery_cost, battery_life_cycles)
        delta = new_profit - best_profit

        if delta > 0 or np.random.rand() < np.exp(delta / temperature):
            actions = proposal
            best_profit = new_profit

        temperature *= cooling_rate

        # if i % 500 == 0:
        #     print(f"Iteration {i}, Profit: €{best_profit:.2f}, Temp: {temperature:.4f}")

    final_profit, final_profits, battery_soc = simulate_with_actions(df_eval, actions, battery_capacity, max_c_rate,
                                                                      efficiency, battery_cost, battery_life_cycles)
    return actions, final_profits, battery_soc, df_eval.index


# -------------------------------
# Plotting
# -------------------------------
def plot_results(timestamps, profit_direct, profit_battery, battery_soc, profit_mcmc, battery_soc_mcmc):
    plt.figure(figsize=(12, 7))

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, profit_direct, label="Direct sale (no battery)")
    plt.plot(timestamps, profit_battery, label="Rule-based battery strategy")
    plt.plot(timestamps, profit_mcmc, label="MCMC-optimized battery strategy")
    plt.ylabel("Cumulative Profit (€)")
    plt.legend()
    plt.title("Comparison of Cumulative Profits")

    plt.subplot(3, 1, 2)
    plt.plot(timestamps, battery_soc, label="SOC (rule-based)")
    plt.plot(timestamps, battery_soc_mcmc, label="SOC (MCMC)")
    plt.ylabel("Battery State of Charge (kWh)")
    plt.legend()
    plt.title("Battery SOC Evolution")

    plt.subplot(3, 1, 3)
    plt.plot(timestamps, np.array(profit_mcmc) - np.array(profit_direct),
             label="Extra Profit (vs. no battery)", color='green')
    plt.ylabel("Profit Gain (€)")
    plt.xlabel("Time")
    plt.legend()
    plt.title("Additional Profit Due to Optimization")

    plt.tight_layout()
    plt.show()


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    price_file = "/home/guillermo/Documentos/ahpc/energy_simulations_git/data/energy-charts_Electricity_production_and_spot_prices_in_Germany_in_2025.csv"
    solar_file = "/home/guillermo/Documentos/ahpc/energy_simulations_git/data/Simulated_Solar_Production.csv"

    df = load_and_prepare_data(price_file, solar_file)
    df_eval = use_real_prices(df)

    # Rule-based baseline
    profit_direct, profit_battery, battery_soc, timestamps = simulate_strategies(df_eval)

    # MCMC optimization
    actions, profit_mcmc, battery_soc_mcmc, _ = optimize_strategy_mcmc(df_eval)

    # Plot all
    plot_results(timestamps, profit_direct, profit_battery, battery_soc, profit_mcmc, battery_soc_mcmc)