#Example code from VitalDB Documentation and from VitalDB Github Examples 

# import pandas as pd

# # Load clinical information data
# df_cases = pd.read_csv("https://api.vitaldb.net/cases")

# # Print the average of death in hospital
# print(df_cases.death_inhosp.mean() * 100)
# print(df_cases.death_inhosp)

# df_cases

import vitaldb 
import pandas as pd 
import matplotlib.pyplot as plt 




#demographics = ["AGE"]

snu = ['SNUADC/ECG_II', 'SNUADC/ART', ]

#caseids = vitaldb.find_cases(['ECG_II', 'ART'])


snu = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/FEM", "SNUADC/CVP" ]
solar = ["Solar8000/VENT_MAWP", "Solar8000/VENT_RR", "Solar8000/VENT_TV", "Solar8000/VENT_PPLAT", "Solar8000/VENT_PIP", "Solar8000/VENT_MV", "Solar8000/VENT_INSP_TM", "Solar8000/BT"]
orch = ["Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP", "Orchestra/RFTN20_CT", "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_VOL"]
bis = ["BIS/BIS", "BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/EMG", "BIS/SEF", "BIS/SQI", "BIS/SR", "BIS/TOTPOW"]


caseids = vitaldb.find_cases(snu+solar+orch+bis)
#caseids = vitaldb.find_cases(demographics)

print(len(caseids))
print(caseids)
#https://github.com/vitaldb/examples/blob/master/eeg_mac.ipynb 
df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # patient information

#df_cases["death_inhosp"] = df_cases["death_inhosp"].astype("bool")
#Count of in-hospital mortality 

df_cases = df_cases[df_cases["caseid"].isin(caseids)]
print(df_cases.head())
print(len(df_cases.index))


#df_cases = df_cases.loc[df_cases["emop"] == 1]
#df_cases = df_cases.loc[df_cases["emop"]]
print(len(df_cases.index))

#print(df_cases["dx"].unique())

#hist = df_cases["dx"].hist()
#print(hist)

#Histogram of diagnosis 
plt.hist(df_cases["airway"])
plt.xticks(rotation=60)
plt.show()

#Mortality - 1
#Preoperative hypertension - 8
#Preopertive diabetes - 5 
#Emergency Op - 7 

#dx - diagnosis 

#deaths = df_cases.loc[df_cases["death_inhosp"] == True & df_cases["caseid"] in caseids]
#print(len(deaths))



#This link has the loading criteria you need: 
#https://github.com/vitaldb/examples/blob/master/eeg_mac.ipynb