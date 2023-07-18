import vitaldb 
import pandas as pd 
import matplotlib.pyplot as plt 

#Help from here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
#And here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
snu = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/FEM", "SNUADC/CVP" ]
bis = ["BIS/BIS", "BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/EMG", "BIS/SEF", "BIS/SQI", "BIS/SR", "BIS/TOTPOW"]
solar = ["Solar8000/VENT_MAWP", "Solar8000/VENT_RR", "Solar8000/VENT_TV", "Solar8000/VENT_PPLAT", "Solar8000/VENT_PIP", "Solar8000/VENT_MV", "Solar8000/VENT_INSP_TM", "Solar8000/BT"]
#43 of these 
valid_cases = [4481, 3719, 1292, 397, 2327, 6297, 5018, 6009, 1820, 2332, 4255, 1191, 1959, 553, 3631, 2738, 818, 1590, 55, 5175, 4283, 5693, 1730, 5442, 3524, 4684, 5837, 1231, 6227, 985, 3930, 2267, 4573, 5983, 2272, 6246, 5607, 1900, 3694, 2168, 1785, 1018, 251]
total_tracks = snu+bis+orch+solar

track_dict = {
    "snu": snu,
    "orch": orch,
    "bis": bis,
    "solar": solar
}


df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # patient information


#Maybe wait until we know exactly what information we want to use here. 