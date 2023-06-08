#Example code from VitalDB Documentation and from VitalDB Github Examples 

# import pandas as pd

# # Load clinical information data
# df_cases = pd.read_csv("https://api.vitaldb.net/cases")

# # Print the average of death in hospital
# print(df_cases.death_inhosp.mean() * 100)
# print(df_cases.death_inhosp)

# df_cases

import vitaldb 


demographics = ["AGE", "SEX", "HEIGHT", "WEIGHT", "BMI", "ASA"]
demographics = ["AGE", "sex"]



#demographics = ["AGE"]

snu = ['SNUADC/ECG_II', 'SNUADC/ART', ]

#caseids = vitaldb.find_cases(['ECG_II', 'ART'])


snu = ['SNUADC/ECG_II', 'SNUADC/ECG_V5', 'SNUADC/ART', "SNUADC/FEM", "SNUADC/CVP" ]
solar = ["Solar8000/VENT_MAWP", "Solar8000/VENT_RR", "Solar8000/VENT_TV", "Solar8000/VENT_PPLAT", "Solar8000/VENT_PIP", "Solar8000/VENT_MV", "Solar8000/VENT_INSP_TM", "Solar8000/BT"]
orch = ["Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP", "Orchestra/RFTN20_CT", "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_VOL"]
bis = ["BIS/BIS", "BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/EMG", "BIS/SEF", "BIS/SQI", "BIS/SR", "BIS/TOTPOW"]



caseids = vitaldb.find_cases(snu+solar+orch+bis)

#caseids = vitaldb.find_cases(demographics)

print(len(caseids))

