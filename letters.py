import json 


def letter_parameter_dicts(phase_name, phase_path_start, letters, input_days, output_days,
 use_scaling_factor, conv=False, prev_phase_base=None,
  delete_stream=None, test=False, deep_lstm=False, 
  deep_ae=False, predict_type=False, transfer_learn=False, 
  transfer_dict=None):
    parameter_dict_list = []
    for letter in letters:
        new_dict = {}
        prev_letter = None
        if prev_phase_base == None:
            prev_phase = phase_name
        else:
            prev_phase = prev_phase_base
        new_dict["delete_stream"] = delete_stream
        new_dict["phase_metrics"] = phase_name+"_"+letter
        new_dict["phase_path"] = phase_path_start+phase_name+"_"+letter+"/"
        new_dict["input_days"] = input_days
        new_dict["output_days"] = output_days
        new_dict["base_dataset_name"] = phase_name+"_"+letter
        new_dict["base_name"] = phase_name+"_"+letter+"_exp"
        new_dict["test"] = test
        new_dict["deep_lstm"] = deep_lstm
        new_dict["deep_ae"] = deep_ae
        new_dict["transfer_learn"] = transfer_learn
        new_dict["transfer_dict"] = transfer_dict
        prev_letter = None 
        if predict_type:
            new_dict["predict_type"] = predict_type
            final_model_type = "time_prediction"
        else:
            final_model_type = "time_regression"
        ae_model_type = "ae"
        #Base 
        if letter == 'A':
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 0]
        elif letter == 'B':
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 1]
        elif letter == 'C':
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [1, 0]
        elif letter == 'D':
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [1, 1]

        #Network 2 - Datastreams 
        elif letter == 'E':
            new_dict["target_model"] = ae_model_type
            new_dict["list_of_base_sets"] = [1, 0]
        elif letter == 'F':
            prev_letter = 'E'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [1, 0]
        elif letter == 'G':
            prev_letter = 'E'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 0]
            new_dict["ae_synthesis"] = "ds"
        elif letter == 'H':
            new_dict["target_model"] = ae_model_type
            new_dict["list_of_base_sets"] = [1, 1]
        elif letter == 'I':
            prev_letter = 'H'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [1, 1]
        elif letter == 'J':
            prev_letter = 'H'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 1]
            new_dict["ae_synthesis"] = "ds"
        #Network 2 - Locations  
        elif letter == 'L':
            new_dict["target_model"] = ae_model_type
            new_dict["list_of_base_sets"] = [0, 1]
        elif letter == 'M':
            prev_letter = 'L'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 1]
        elif letter == 'N':
            prev_letter = 'L'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 0]
            new_dict["ae_synthesis"] = "l"
        #O and P really already taken care of technically 
        # elif letter == 'O':
        #     new_dict["target_model"] = "ae"
        #     new_dict["list_of_base_sets"] = [1, 1]
        # elif letter == 'P':
        #     prev_letter = 'O'
        #     new_dict["target_model"] = "time_regression"
        #     new_dict["list_of_base_sets"] = [1, 1]
        elif letter == 'Q':
            prev_letter = 'H'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [1, 0]
            new_dict["ae_synthesis"] = "l"
        
        #Network 3 - Datastreams 
        if letter == 'S':
            prev_letter = 'E'
            new_dict["target_model"] = ae_model_type
            new_dict["list_of_base_sets"] = [0, 0]
            new_dict["ae_synthesis"] = "ds"
        if letter == 'T':
            prev_letter = 'S'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 0]
        if letter == 'U':
            prev_letter = 'H'
            new_dict["target_model"] = ae_model_type
            new_dict["list_of_base_sets"] = [0, 1]
            new_dict["ae_synthesis"] = "ds"
        if letter == 'V':
            prev_letter = 'U'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 1]
        if letter == 'W':
            prev_letter = 'U'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 0]
            new_dict["ae_synthesis"] = "l"
        #Network 3 - Locations  
        if letter == 'X':
            prev_letter = 'L'
            new_dict["target_model"] = ae_model_type
            new_dict["list_of_base_sets"] = [0, 0]
            new_dict["ae_synthesis"] = "l"
        if letter == 'Y':
            prev_letter = 'X'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 0]
        if letter == 'Z':
            prev_letter = 'H'
            new_dict["target_model"] = ae_model_type
            new_dict["list_of_base_sets"] = [1, 0]
            new_dict["ae_synthesis"] = "l"
        if letter == 'AA':
            prev_letter = 'Z'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [1, 0]
        if letter == 'AB':
            prev_letter = 'Z'
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 0]
            new_dict["ae_synthesis"] = "ds"
        #Network 4 - All  
        if letter == 'AC':
            new_dict["target_model"] = ae_model_type
            new_dict["list_of_base_sets"] = [0, 0]
        if letter == 'AD':
            prev_letter = "AC"
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 0]
        #Network 4 - All datastreams, one location 
        if letter == "AE":
            new_dict["target_model"] = ae_model_type
            new_dict["list_of_base_sets"] = [0, 1]
        if letter == 'AF':
            prev_letter = "AE"
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 1]
        if letter == 'AG':
            prev_letter = "AE"
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 0]
            new_dict["ae_synthesis"] = "l"
        #Network 4- All locations, one datastream 
        if letter == "AH":
            new_dict["target_model"] = ae_model_type
            new_dict["list_of_base_sets"] = [1, 0]
        if letter == 'AI':
            prev_letter = "AH"
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [1, 0]
        if letter == 'AJ':
            prev_letter = "AH"
            new_dict["target_model"] = final_model_type
            new_dict["list_of_base_sets"] = [0, 0]
            new_dict["ae_synthesis"] = "ds"
        #End 
        if new_dict["target_model"] == "ae":
            #Keep for phases 6 and 7!
            #new_dict["scaling_factors"] = [0.3, 0.5, 0.7]
            #Otherwise stick with phases from now on 
            new_dict["scaling_factors"] = [0.7]
        else:
            new_dict["scaling_factors"] = [8, 32, 64]
        if prev_letter != None:
            new_dict["use_scaling_factor"] = use_scaling_factor
            new_dict["ae_letter"] = prev_letter
            new_dict["ae_phase"] = prev_phase
            model_name = prev_phase+"_"+prev_letter+"_exp"+str(use_scaling_factor)
            new_dict["ae_models"]= [model_name]
            prev_dataset_name = prev_phase+"_"+prev_letter
            new_dict["ae_prev_names"]=  [prev_dataset_name]
        if conv == True:
            new_dict["conv"] = True
        else:
            new_dict["conv"] = False
        if conv == True and "ae_prev_names" in list(new_dict.keys()):
            new_dict["conv_and_prev_ae"] = True
        else:
            new_dict["conv_and_prev_ae"] = False
        new_dict["letter"] = letter
        parameter_dict_list.append(new_dict.copy())
        new_dict = {}


    print(json.dumps(parameter_dict_list, indent=4))

    #NOTE - You need to make sure K and L work! 

    #RUN THEM HERE 
    for parameters_dict in parameter_dict_list:
        print(f"Phase {phase_name} Letter: {parameters_dict['phase_metrics']}")
        #Generate descriptors 
        descriptors_list = ddl.run_generate(parameters_dict)
        print(json.dumps(descriptors_list[0], indent=3))
        #Save the list 
        ddl.save_list(parameters_dict, descriptors_list)
        #If it's a test, just run that 
        if test == True:
            print("Running Test")
            indexes = [0]
            experiment_1 = return_test_experiment(descriptors_list)
            ddl.run_test(indexes, experiment_1, descriptors_list)
        else:
            #Make the datasets
            marl.make_datasets(parameters_dict["phase_path"])
            #Make the experiment descriptors
            experiments = edl.run_generate(parameters_dict)
            edl.save_list(parameters_dict, experiments)
            #Run the experiments 
            marl.run_experiments(parameters_dict["phase_path"])
        