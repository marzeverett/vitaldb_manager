#Example code from VitalDB Documentation and from VitalDB Github Examples 

import pandas as pd

# Load clinical information data
df_cases = pd.read_csv("https://api.vitaldb.net/cases")

# Print the average of death in hospital
print(df_cases.death_inhosp.mean() * 100)
print(df_cases.death_inhosp)

df_cases
