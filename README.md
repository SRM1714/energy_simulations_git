Energy Optimization and Decision Simulation Framework
This repository presents a simulation framework designed to explore and optimize energy-related decision-making processes based on photovoltaic (PV) production, energy demand, battery storage, and market behavior under various scenarios.

The project is motivated by the operational dynamics of the electricity markets in Spain, Italy, Germany, and Switzerland. For the accurate representation of these markets, several key academic and industry papers have been reviewed. You can find the links to those papers

Repository Structure
The repository is organized into several folders and scripts:

data/ — contains raw and processed datasets used in model training and simulation

outputs/ — includes example outputs from simulations

code/ — houses the core simulation and modeling scripts

Core Modules Overview
1. predict_production.py
This script is used to predict hourly solar photovoltaic (PV) production using a variety of models. Among the tested algorithms are Artificial Neural Networks (ANNs), Random Forests, XGBoost, and others.

Input features used for production forecasting include:

Ambient temperature

Module temperature

Solar irradiance

Time of day

Panel surface area

The model is trained on historical weather and generation data to accurately capture production behavior across time.

2. predict_price.py
This module predicts hourly electricity prices, based on factors such as:

Day of the week

Hour of the day

Temperature

Forecasted demand

Renewable generation share

The price models help simulate realistic market conditions for optimizing energy dispatch strategies.

3. simulation.py
This is the heart of the decision-making engine. It evaluates different energy dispatch strategies:

Rule-based strategy: simple heuristic charging/discharging logic

MCMC-based strategy: optimized control policy using Markov Chain Monte Carlo (MCMC) methods

The script simulates how decisions around battery charging, discharging, and direct energy sales to the grid affect cumulative profit over time.

4. self_sufficiency_simulation.py
This script runs a Monte Carlo simulation to evaluate the likelihood of a household being fully energy self-sufficient under a given PV system and battery configuration.

It considers:

Randomized solar weather conditions

Realistic demand profiles

Storage dynamics

The output shows in how many scenarios the household could meet its full energy demand without relying on the external grid.

5. montecarlo_simulation.py
In this script, we evaluate long-term profitability using our full decision engine. By combining:

The price prediction model

The PV production model

And the decision-making strategies

we run multiple Monte Carlo simulations to assess:

The expected profit of each strategy over time

How our optimized algorithm compares to naive approaches (like direct sale or rule-based storage)


