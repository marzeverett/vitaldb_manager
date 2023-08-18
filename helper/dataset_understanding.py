import vitaldb 
import pandas as pd 
import matplotlib.pyplot as plt 

#Need to sys path append here 

#Help from here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
#And here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
snu = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/FEM", "SNUADC/CVP" ]
orch = ["Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP", "Orchestra/RFTN20_CT", "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_VOL"]
bis = ["BIS/BIS", "BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/EMG", "BIS/SEF", "BIS/SQI", "BIS/SR", "BIS/TOTPOW"]
solar = ["Solar8000/VENT_MAWP", "Solar8000/VENT_RR", "Solar8000/VENT_TV", "Solar8000/VENT_PPLAT", "Solar8000/VENT_PIP", "Solar8000/VENT_MV", "Solar8000/VENT_INSP_TM", "Solar8000/BT"]
#43 of these 
valid_cases = [4481, 3719, 1292, 2327, 5018, 6009, 1820, 2332, 4255, 1191, 1959, 553, 3631, 2738, 818, 1590, 55, 4283, 5693, 5442, 3524, 4684, 5837, 1231, 6227, 985, 3930, 2267, 4573, 5983, 2272, 6246, 5607, 1900, 3694, 1785, 1018, 251]
clinical_info = ["anestart", "aneend", "age", "sex", "height", "weight", 
"bmi", "emop", "dx", "dis"]

cat = ["sex", "dx"]

total_tracks = snu+bis+orch+solar

#Read in the cases 
df_cases = pd.read_csv("../vital_csvs/clinical_info.csv")


#Restrict to only the cases of interest 
df_cases = df_cases[df_cases["caseid"].isin(valid_cases)]

print(len(df_cases))

# #Histogram - useful info 
# param = "emop"
# sub_df = df_cases[param]
# #sub_df = sub_df / 86400
# plt.hist(sub_df)
# plt.xticks(rotation=10)
# plt.xlabel("Count")
# plt.ylabel(param)
# plt.title(f"Histogram of {param} in Dataset")
# plt.show()


#print(df_cases['dx'].value_counts())


#df_cases = df_cases.loc[df_cases["emop"] == 1]
# # #df_cases = df_cases.loc[df_cases["emop"]]
# # ##
#print(len(df_cases.index))
# print(df_cases["caseid"])

#print(df_cases["dx"].unique())


# #Figure out number of observations 
# case_ids = []
# obs = []
# for case in valid_cases:
#     #Read in the case
#     df = pd.read_csv(f"../vital_csvs/{case}.csv")
#     obs.append(len(df.index))
#     case_ids.append(case)
#     # df = df.loc[df["dis_mortality_risk"] == 1]
#     # print(len(df.index))

# print(case_ids)

# print("AVG ", sum(obs)/len(obs))
# print("MAX ", max(obs), case_ids[obs.index(max(obs))])
# print("MIN ", min(obs), case_ids[obs.index(min(obs))])


# #Figure out number of observations 
# case_ids = []
# obs = []
# null_cases = 0 
# for case in valid_cases:
#     #Read in the case
#     df = pd.read_csv(f"../vital_csvs/{case}_normalized.csv")
#     if df.isnull().values.any():
#         case_ids.append(case)
#     # df = df.loc[df["dis_mortality_risk"] == 1]
#     # print(len(df.index))

# print(case_ids)


# #Figure out number of gluc and dis risk  
# case_ids = []
# obs = []
# gluc_risk = []
# for case in valid_cases:
#     #Read in the case
#     df = pd.read_csv(f"../vital_csvs/{case}.csv")
#     df_1 = df.loc[df["dis_mortality_risk"] == 1]
    
#     if "gluc_risk" in list(df.columns):
#         df_2 = df.loc[df["gluc_risk"] == 1]
#         if len(df_2.index) > 0:
#             gluc_risk.append(case)
#     if len(df_1.index) > 0:
#         obs.append(case)

# print("Gluc risk")
# print(len(gluc_risk))
# print(gluc_risk)
# print("Dis Risk")
# print(len(obs))
# print(obs)




