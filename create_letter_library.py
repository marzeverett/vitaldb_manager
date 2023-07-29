import json 


#End Target Model
lstm_letters = ["A", "B", "C", "D", "F", "G", "I", "J",
                "M", "N", "Q", "T", "V", "W", "Y", "AA", "AD"]
ae_letters = ["E", "H", "L", "S", "U", "X", "Z", "AC"]

#By datastream/location separation 
no_separation_letters = ["A", "G", "N", "S", "T", "W", "X", "Y", "AC","AD"]
separate_by_datastream_letters = ["C", "E", "F", "Q", "Z", "AA"]
separate_by_location_letters = ["B", "L", "J", "M", "U", "V"]
separate_by_location_and_datastream_letters = ['D',"H",'I']

network_1_letters = ['A', 'B', 'C', 'D']
network_2_letters = ['E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Q']
network_3_letters = ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA']
network_4_letters = ['F', 'M', 'AD']

#This keys will always be in the dictionary 
def return_default_param_dict():
    default_param_dict = {
        "phase_name": "test",
        "phase_path_start": "generated_files/",
        "delete_stream": False,
        "use_prev_ae_phase": False,
        "scaling_factor": 0.7,
        "task_type": "regression",
        "model_group": "0",
        "input_samples": [30],
        "output_samples": [10],
        "test": False,
        "retrain": False,
        "predict_type": None,
        "building_on_phase": False,
        "retrain_dict": {
            "retrain_from_phase": "0",
            "separation_scheme": "ds",
            "retrained_letters": ["E", "H"]
        },
        "ae_letter": None,
        "ae_synthesis": None
    }
    return default_param_dict.copy()

#Take in phase dict and letters, return list of parameter dicts for data
#Descriptors 
def return_letter_parameter_dict_list(phase_dict, letters):
    parameter_dict_list = []
    #For each letter in the phase 
    for letter in letters:
        #First get the default parameter dict
        param_dict = {}
        param_dict = return_default_param_dict()
        param_dict.update(phase_dict.copy())
        param_dict["phase_metrics"] = param_dict["phase_name"]+"_"+letter
        param_dict["phase_path"] = param_dict["phase_path_start"]+param_dict["phase_name"]+"_"+letter+"/"
        param_dict["base_dataset_name"] = param_dict["phase_name"]+"_"+letter
        param_dict["experiment_base"] = param_dict["phase_name"]+"_"+letter

        #Assign the LSTM model types 
        if letter in lstm_letters:
            param_dict["target_model"] = "lstm"
            param_dict["model_iterations"] = [8, 32, 64]
            param_dict["scaling_factors"] = param_dict["lstm_scaling_factors"]
        #Assign the AE model types 
        elif letter in ae_letters:
            param_dict["target_model"] = "ae"
            param_dict["model_iterations"] = [0.7]
            param_dict["scaling_factors"] = param_dict["ae_scaling_factors"]

        #Assign the Non-separation models
        if letter in no_separation_letters:
            param_dict["ds_scheme"] = 0
            param_dict["l_scheme"] = 0
            
        #Assign the models separated by datastream
        if letter in separate_by_datastream_letters:
            param_dict["ds_scheme"] = 1
            param_dict["l_scheme"] = 0

        #Assign the models separated by location 
        if letter in separate_by_location_letters:
            param_dict["ds_scheme"] = 0
            param_dict["l_scheme"] = 1

        #Assign the models separated by location and datastream 
        if letter in separate_by_location_and_datastream_letters:
            param_dict["ds_scheme"] = 1
            param_dict["l_scheme"] = 1

        #Network 2 - Datastreams 
        elif letter == 'G':
            param_dict["ae_letter"] = 'E'
            param_dict["ae_synthesis"] = "ds"
        elif letter == 'I':
            param_dict["ae_letter"] = 'H'
        elif letter == 'J':
            param_dict["ae_letter"] = 'H'
            param_dict["ae_synthesis"] = "ds"
        #Network 2 - Locations  
        elif letter == 'M':
            param_dict["ae_letter"] = 'L'
        elif letter == 'N':
            param_dict["ae_letter"] = 'L'
            param_dict["ae_synthesis"] = "l"
        elif letter == 'Q':
            param_dict["ae_letter"] = 'H'
            param_dict["ae_synthesis"] = "l"
        #Network 3 - Datastreams 
        if letter == 'S':
            param_dict["ae_letter"] = 'E'
            param_dict["ae_synthesis"] = "ds"
        if letter == 'T':
            param_dict["ae_letter"] = 'S'
        if letter == 'U':
            param_dict["ae_letter"] = 'H'
            param_dict["ae_synthesis"] = "ds"
        if letter == 'V':
            param_dict["ae_letter"] = 'U'
        if letter == 'W':
            param_dict["ae_letter"] = 'U'
            param_dict["ae_synthesis"] = "l"
        #Network 3 - Locations  
        if letter == 'X':
            param_dict["ae_letter"] = 'L'
            param_dict["ae_synthesis"] = "l"
        if letter == 'Y':
            param_dict["ae_letter"] = 'X'
        if letter == 'Z':
            param_dict["ae_letter"] = 'H'
            param_dict["ae_synthesis"] = "l"
        if letter == 'AA':
            param_dict["ae_letter"] = 'Z'
        if letter == 'AB':
            param_dict["ae_letter"] = 'Z'
            param_dict["ae_synthesis"] = "ds"
        if letter == 'AD':
            param_dict["ae_letter"] = "AC"
        param_dict["letter"] = letter
        parameter_dict_list.append(param_dict.copy())


    #print(json.dumps(parameter_dict_list, indent=4))
    return parameter_dict_list


    # #RUN THEM HERE 
    # for parameters_dict in parameter_dict_list:
    #     print(f"Phase {phase_name} Letter: {parameters_dict['phase_metrics']}")
    #     #Generate descriptors 
    #     descriptors_list = ddl.run_generate(parameters_dict)
    #     print(json.dumps(descriptors_list[0], indent=3))
    #     #Save the list 
    #     ddl.save_list(parameters_dict, descriptors_list)
    #     #If it's a test, just run that 
    #     if test == True:
    #         print("Running Test")
    #         indexes = [0]
    #         experiment_1 = return_test_experiment(descriptors_list)
    #         ddl.run_test(indexes, experiment_1, descriptors_list)
    #     else:
    #         #Make the datasets
    #         marl.make_datasets(parameters_dict["phase_path"])
    #         #Make the experiment descriptors
    #         experiments = edl.run_generate(parameters_dict)
    #         edl.save_list(parameters_dict, experiments)
    #         #Run the experiments 
    #         marl.run_experiments(parameters_dict["phase_path"])
        