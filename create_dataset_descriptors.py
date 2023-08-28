import json 
import itertools 

#Help getting cartesian product: https://note.nkmk.me/en/python-itertools-product/ 

cases = {
    4481: {"index": 0},
    3719: {"index": 1},
    1292: {"index": 2},
    2327:{"index": 3},
    5018: {"index": 4},
    6009: {"index": 5},
    1820: {"index": 6},
    2332: {"index": 7},
    4255: {"index": 8},
    1191: {"index": 9},
    1959: {"index": 10},
    553: {"index": 11},
    3631: {"index": 12},
    2738: {"index": 13},
    818: {"index": 14},
    1590: {"index": 15},
    55: {"index": 16},
    4283: {"index": 17},
    5693: {"index": 18},
    5442: {"index": 19},
    3524: {"index": 20},
    4684: {"index": 21},
    5837: {"index": 22},
    1231: {"index": 23},
    6227: {"index": 24},
    985: {"index": 25},
    3930: {"index": 26},
    2267: {"index": 27},
    4573: {"index": 28},
    5983: {"index": 29},
    2272: {"index": 30},
    6246: {"index": 31},
    5607: {"index": 32},
    1900: {"index": 33},
    3694: {"index": 34},
    1785: {"index": 35},
    1018: {"index": 36},
    251: {"index": 37}
}

#CHANGE HERE 
#Change here! Need to make normal later 
cases = {
    #4481: {"index": 0},
    3719: {"index": 1},
    818: {"index": 14}
}

datastreams = {
    "orch": {
        "index": 1,
        "fields": ["Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP", "Orchestra/RFTN20_CT", "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_VOL"]
        },
    "snu": {
        "index": 2,
        "fields": ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/FEM", "SNUADC/CVP" ]
        },
    "solar": {
        "index": 3,
        "fields": ["Solar8000/VENT_MAWP", "Solar8000/VENT_RR", "Solar8000/VENT_TV", "Solar8000/VENT_PPLAT", "Solar8000/VENT_PIP", "Solar8000/VENT_MV", "Solar8000/VENT_INSP_TM", "Solar8000/BT"]
        },
    # "clinical": {
    #     "index": 4,
    #     "fields": ["anestart", "aneend", "age", "sex", "height", "weight",
    #                 "bmi", "dx", "dis", "preop_pft", "preop_plt", "preop_pt", 
    #                 "preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_cr", 
    #                 "intraop_uo", "intraop_ffp"]
    # },
    # "lab": {
    #     "index": 5,
    #     "fields": ["wbc", "hb", "hct", "gluc", "cr", "na", "k", "ammo",
    #             "ptinr", "pt%", "ptsec", "aptt", "ph"]
    # }
}

clinical_fields =  ["anestart", "aneend", "age", "sex", "height", "weight",
                     "bmi", "dx", "dis", "preop_pft", "preop_plt", "preop_pt", 
                     "preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_cr", 
                     "intraop_uo", "intraop_ffp"]

predictions = ["emop", "dis_mortality_risk", "gluc_risk"]


#End Target Model
lstm_letters = ["A", "B", "C", "D", "F", "G", "I", "J",
                "M", "N", "Q", "T", "V", "W", "Y", "AA", "AD"]
ae_letters = ["E", "H", "L", "S", "U", "X", "Z", "AC"]
#By datastream/location separation 
no_separation_letters = ["A", "G", "N", "T", "W", "Y", "AD"]
separate_by_datastream_letters = ["C", "F", "Q", "AA"]
separate_by_location_letters = ["B", "J", "M", "V"]
separate_by_location_and_datastream_letters = ['D', 'I']
#By Network type 
network_1_letters = ['A', 'B', 'C', 'D']
network_2_letters = ['E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Q']
network_3_letters = ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA']
network_4_letters = ['F', 'M', 'AD']


def return_default_dataset_descriptor():
    default_dataset_descriptor = {
        "output_offset": 1,
        "l_combo": 0,
        "keys": ["Time", "caseid"]
        #"keys": ["Time"]
        #"location_scheme": 0,
    }
    return default_dataset_descriptor.copy()

#Dataset Descriptor Sets 
#Mainly need to figure out what AE paths are needed 

#Here we just need to break up by datastream, NOT by location. 
#You might see if this is worth copying your code and changing like two things... 
#Since the only thing you really need to do is NOT merge the dataset 
def create_dataset_name(phase, letter, ds_scheme, ds_combo, dataset_descriptor):
    dataset_name = f"{phase}_{letter}_{dataset_descriptor['model_group']}_{ds_scheme}_{dataset_descriptor['l_scheme']}_{ds_combo}_{dataset_descriptor['l_combo']}_{dataset_descriptor['input_samples']}_{dataset_descriptor['output_samples']}"
    return dataset_name

def create_dataset_class(dataset_descriptor):
    dataset_class_dict = {
        "phase": dataset_descriptor["phase_name"],
        "letter": dataset_descriptor["letter"],
        "model_group": dataset_descriptor["model_group"],
        "ds_scheme": dataset_descriptor["ds_scheme"],
        "l_scheme": dataset_descriptor["l_scheme"],
        "ds_combo": dataset_descriptor["ds_combo"],
        "l_combo": dataset_descriptor["l_combo"],
        "input_samples": dataset_descriptor["input_samples"],
        "output_samples": dataset_descriptor["output_samples"]
    }
    return dataset_class_dict


#ETC -- start here! 
def create_ae_path_dict(dataset_descriptor, phase, ds_scheme, ds_combo):
    ae_dataset_name = create_dataset_name(phase, dataset_descriptor["ae_letter"], ds_scheme, ds_combo, dataset_descriptor)
    ae_path_dict = {
        "phase": phase,
        "letter": dataset_descriptor["ae_letter"],
        "dataset": ae_dataset_name
    }
    return ae_path_dict


#Need to gatekeep for retraining as well 
#List of ae model paths 
def get_ae_paths(dataset_descriptor, using_datastreams):
    #By default, we assume we are training with our own 
    use_phase = dataset_descriptor["phase_name"]
    use_ds_scheme = dataset_descriptor["ds_scheme"]
    ae_paths_dict_list = []
    #If this isn't false, we assume we are using a previous phase  
    if dataset_descriptor["building_on_phase"]:
        use_phase = dataset_descriptor["building_on_phase"]
    #If we are synthesizing on datastream:
    if dataset_descriptor["ae_synthesis"] == "ds":
        #Get each datastream that synthesizes
        use_ds_scheme = 1 
        for datastream_name in list(using_datastreams.keys()):
            synth_datastream_index = using_datastreams[datastream_name]["index"]
            new_use_phase = use_phase   
            #If we are retraining, on an affected letter, and an affected datastream 
            if dataset_descriptor["retrain"]:
                if dataset_descriptor["letter"] in dataset_descriptor["retrain_dict"]["retrained_letters"]:
                    if dataset_descriptor["ds_combo"] != synth_datastream_index:
                        new_use_phase = dataset_descriptor["retrain_from_phase"]

            #Make the path, swapping out the datastream scheme 
            ae_dict = create_ae_path_dict(dataset_descriptor, new_use_phase, use_ds_scheme, synth_datastream_index)
            ae_paths_dict_list.append(ae_dict)
    else:
        #Just make the path 
        ae_dict = create_ae_path_dict(dataset_descriptor, use_phase, use_ds_scheme, dataset_descriptor["ds_combo"])
        ae_paths_dict_list.append(ae_dict)
    return ae_paths_dict_list


def generate_dataset_descriptor(dataset_descriptor, datastream_index, input_sample, output_sample, using_datastreams):
    #Put in the cases:
    dataset_descriptor["datasets"] = list(cases.keys())
    
    #Get your input fields - either all or broken up by datastream
    if datastream_index == "ALL":
        input_fields = []
        for datastream_name in list(using_datastreams.keys()):
            input_fields = input_fields + using_datastreams[datastream_name]["fields"]
            ds_combo = 0
    else:
        input_fields = using_datastreams[datastream_index]["fields"]
        ds_combo = using_datastreams[datastream_index]["index"]
    dataset_descriptor["input_fields"] = input_fields
    dataset_descriptor["ds_combo"] = ds_combo

    #Iput and output samples
    dataset_descriptor["input_samples"] = input_sample
    dataset_descriptor["output_samples"] = output_sample
    
    #Get your output fields - either the same as input fields or different, prediction
    if dataset_descriptor["target_model"] == "ae":
        dataset_descriptor["output_fields"] = input_fields
    elif  dataset_descriptor["task_type"] == "regression":
        #Right here - need to take out any clinical data. (for regression, but not ae?)
        #Maybe for both. 
        dataset_descriptor["output_fields"] = input_fields
    
    elif dataset_descriptor["task_type"] == "prediction":
        if isinstance(dataset_descriptor["predict_type"], list):
            dataset_descriptor["output_fields"] = dataset_descriptor["predict_type"]
        else:
            dataset_descriptor["output_fields"] = [dataset_descriptor["predict_type"]]
    
    #Right here -- can add input clinical data to input fields, ONLY for regression
    #If it's not an autoencoder AND not prediction 
    if dataset_descriptor["target_model"] != "ae":
        if dataset_descriptor["task_type"] != "prediction":
            dataset_descriptor["input_fields"] = dataset_descriptor["input_fields"]+clinical_fields

    #These depend on ae_synthesis, prev phase, or retrain
    this_letter = dataset_descriptor['letter']
    dataset_descriptor["ae_dicts"] = []
    if "ae_letter" in list(dataset_descriptor.keys()):
        if dataset_descriptor["ae_letter"] != None:
            dataset_descriptor["ae_dicts"] = get_ae_paths(dataset_descriptor, using_datastreams)

    dataset_descriptor["dataset_name"] = create_dataset_name(dataset_descriptor["phase_name"], dataset_descriptor["letter"], dataset_descriptor["ds_scheme"], dataset_descriptor["ds_combo"], dataset_descriptor)
    dataset_descriptor["dataset_class"] = create_dataset_class(dataset_descriptor)
    return dataset_descriptor
    


def break_out_descriptors(dataset_descriptor):
    base_descriptor = dataset_descriptor.copy()
    #Get all different types of input/output - sub_dicts 
    #All or individual 
    #Factoring in the delete stream. 
    using_datastreams = datastreams.copy()
    #If we need to delete a stream, pop it out of the datastreams keys 
    if dataset_descriptor["delete_stream"]:
        for stream in dataset_descriptor["delete_stream"]:
            using_datastreams.pop(stream)

    #If it's broken out by datastream - going to have to pass the using datastreams 
    if dataset_descriptor["ds_scheme"] == 1:
        input_streams = list(using_datastreams.keys())
    else:
        #Keyword - know to use all input streams 
        input_streams = ["ALL"]
    #Create a list of datastream index, input samples, and output samples 
    #Using cross product 
    input_samples = dataset_descriptor["input_samples"]
    output_samples = dataset_descriptor["output_samples"]
    relevant_lists = [input_streams, input_samples, output_samples]
    descriptors_list = []
    #Generate the descriptors
    for datastream_index, input_sample, output_sample in itertools.product(input_streams, input_samples, output_samples):
        new_descriptor = generate_dataset_descriptor(dataset_descriptor.copy(), datastream_index, input_sample, output_sample, using_datastreams)
        #Here is where we check for retrain!!
        if new_descriptor["retrain"]:
            retrain_dict = dataset_descriptor["retrain_dict"]
            #If it's in the letters, only add if it's on the appropriate stream
            if new_descriptor["letter"] in retrain_dict["retrained_letters"]:
                prev_index = using_datastreams[retrain_dict["prev_delete_stream"]]["index"]
                if dataset_descriptor["ds_combo"] == prev_index:
                    descriptors_list.append(new_descriptor)
        #If it's not a retrain, add it
        else:
            descriptors_list.append(new_descriptor)
    return descriptors_list


#Regression, AE, or predict 
#This will get a parameter dict from a particular letter 
def create_dataset_descriptor_list_from_parameter_dict(parameter_dict):
    dataset_descriptor = return_default_dataset_descriptor()
    dataset_descriptor.update(parameter_dict)
    dataset_descriptor_list = break_out_descriptors(dataset_descriptor)
    return dataset_descriptor_list
    

#Test and see if this worked! -- Make sure you test with all the letters. 


