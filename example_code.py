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

#Help from here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
#And here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 





#demographics = ["AGE"]

#snu = ['SNUADC/ECG_II', 'SNUADC/ART', ]

#caseids = vitaldb.find_cases(['ECG_II', 'ART'])


snu = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/FEM", "SNUADC/CVP" ]
orch = ["Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP", "Orchestra/RFTN20_CT", "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_VOL"]
bis = ["BIS/BIS", "BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/EMG", "BIS/SEF", "BIS/SQI", "BIS/SR", "BIS/TOTPOW"]
solar = ["Solar8000/VENT_MAWP", "Solar8000/VENT_RR", "Solar8000/VENT_TV", "Solar8000/VENT_PPLAT", "Solar8000/VENT_PIP", "Solar8000/VENT_MV", "Solar8000/VENT_INSP_TM", "Solar8000/BT"]


caseids = vitaldb.find_cases(snu+solar+orch+bis)
#caseids = vitaldb.find_cases(demographics)

print(len(caseids))
print(caseids)
#https://github.com/vitaldb/examples/blob/master/eeg_mac.ipynb 
df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # patient information

#df_cases.to_csv("vital_csvs/clinical_info.csv")

#df_cases["death_inhosp"] = df_cases["death_inhosp"].astype("bool")
#Count of in-hospital mortality 

df_cases = df_cases[df_cases["caseid"].isin(caseids)]
##
#print(df_cases.head())
#print(len(df_cases.index))
#total_tracks = snu+solar+orch+bis 
total_tracks = snu+bis+orch+solar

# print("Make VitalFile")
# vf = vitaldb.VitalFile(caseids[0], total_tracks)

# print("Make Dataframe")
# df = vf.to_pandas(total_tracks, 1)

# print("Save Dataframe")
# df.to_csv("vital_first.csv")

def trim_df(df, columns):
    start_indexes = []
    end_indexes = []
    for column in columns:
        #find first valid index and append it 
        first = df[column].first_valid_index()
        start_indexes.append(first)
        #find last valid index and append it 
        last = df[column].last_valid_index()
        end_indexes.append(last)
    #Take max start index
    max_val = max(start_indexes)
    min_val = min(end_indexes)
    #Take min end index
    print(max_val, min_val)
    df = df.iloc[max_val:min_val]
    return df 

def fill_gaps(df):
    df = df.fillna(method="ffill")
    return df 


def add_static_info(case_id, df):
    df_cases = pd.read_csv("vital_csvs/clinical_info.csv")
    df_case = df_cases.loc[df_cases["caseid"] == case_id]
    case_columns = df_case.columns.values.tolist()
    for col in case_columns:
        df[col] = df_case[col].item()
    return df 

# folder = "vital_csvs/"
# #caseids = [caseids[0]]
# for caseid in caseids:
#     print("Make VitalFile")
#     vf = vitaldb.VitalFile(caseid, total_tracks)
#     print("Make Dataframe")
#     df = vf.to_pandas(total_tracks, 1)
#     print("Save Dataframe")
#     #process it 
#     df= trim_df(df, total_tracks)
#     df = fill_gaps(df)
#     df = add_static_info(caseid, df)

#     save_name = f"{folder}{caseid}.csv"
#     df.to_csv(save_name)


#df_cases = df_cases.loc[df_cases["emop"] == 1]
#df_cases = df_cases.loc[df_cases["emop"]]
##
#print(len(df_cases.index))

#print(df_cases["dx"].unique())

#hist = df_cases["dx"].hist()
#print(hist)

#Histogram - useful info 
# #Histogram of diagnosis 
# plt.hist(df_cases["position"])
# plt.xticks(rotation=10)
# plt.show()

#Mortality - 1
#Preoperative hypertension - 8
#Preoperative diabetes - 5 
#Emergency Op - 7 '
#optype = transplantations (all)
#Multiple diagnosis 
#Mostly older patients 
#Preoperative bun might be good to predict as well 

#dx - diagnosis 

#Demographic info to include
#age
#sex
#height
#weight
#bmi
#asa


#deaths = df_cases.loc[df_cases["death_inhosp"] == True & df_cases["caseid"] in caseids]
#print(len(deaths))



#This link has the loading criteria you need: 
#https://github.com/vitaldb/examples/blob/master/eeg_mac.ipynb


#Here's what we'll do: 
#First, download the data for each case
#Look at the first first valid for each column value
#Look at the last valid value for each column 
#Trim the case to just those
#Forward fill
#
#Will go into a folder case/ingo