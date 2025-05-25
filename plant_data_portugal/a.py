import pandas as pd

# Read the Excel file
df = pd.read_excel('/home/guillermo/Documentos/ahpc/energy_simulations_git/plant_data_portugal/PV Plants Datasets.xlsx')

# Write to CSV without the index column
df.to_csv('/home/guillermo/Documentos/ahpc/energy_simulations_git/plant_data_portugal/PV_Plants_Datasets.csv', index=False)