import json 
import os 
import pickle 
import create_letter_library
import create_dataset_descriptors
import create_dataset_library 
import create_experiment_descriptors 
import create_model 
import create_metrics 

#Save Paths -- for reference 
# generated_files/descriptors/phase_letter/{files} (Do we bother?)
# generated_files/datasets/phase_letter/dataset_name/{files}
# generated_files/experiments/phase_letter/model/dataset_name/{files}
# generated_files/experiments/metrics/

def save_descriptor_and_result(dataset_descriptor, dataset_result):
    phase = dataset_descriptor["phase_name"]
    phase_letter = dataset_descriptor["letter"]
    dataset_name = dataset_descriptor["dataset_name"]
    base_path = f"generated_files/datasets/{phase}_{phase_letter}/{dataset_name}/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    descriptors_path = base_path+"dataset_descriptor.pickle"
    result_path = base_path+"dataset_result.pickle"
    #Save descriptors 
    with open(descriptors_path, "wb") as f:
        pickle.dump(dataset_descriptor, f)
    #Save dataset result
    with open(result_path, "wb") as f:
        pickle.dump(dataset_result, f)


#Actually run the experiments 
def run_phase_experiments(phase_dict, letters):
    #first, get each parameter dicts for the letters in the experiment
    #There is one parameter dict for every letter in the  
    parameter_dict_list = create_letter_library.return_letter_parameter_dict_list(phase_dict, letters)
    #For each letter in our slate:
    for param_dict in parameter_dict_list:
        #Get all the dataset descriptors for that letter 
        dataset_descriptors_list = create_dataset_descriptors.create_dataset_descriptor_list_from_parameter_dict(param_dict)        
        #For every dataset descriptor we generated:
        print(json.dumps(param_dict, indent=4)) 
        #print(f"{len(dataset_descriptors_list)} number of descriptors generated")
        for single_dataset_descriptor in dataset_descriptors_list:
            #create a dataset result
            dataset_result = create_dataset_library.create_dataset_from_dataset_descriptor(single_dataset_descriptor)
            #Save the descriptor and result into the correct folder
            save_descriptor_and_result(single_dataset_descriptor, dataset_result)
            #For each scaling factor (experimental model modification) in a given dataset descriptor:
            for scaling_factor in single_dataset_descriptor["scaling_factors"]:
                experiment_descriptor = {}
                dataset_result = {}
                experiment_object = {}
                experiment_result = {}
                #We create that experiment descriptor 
                experiment_descriptor = create_experiment_descriptors.create_experiment_descriptor(scaling_factor, single_dataset_descriptor, dataset_result)
                #Run the experiment 
                single_dataset_descriptor, dataset_result, experiment_object, experiment_result = create_model.experiment_from_experiment_object(single_dataset_descriptor, dataset_result, experiment_descriptor)
                #Create the metrics 
                create_metrics.visualize_and_analyze(single_dataset_descriptor, dataset_result, experiment_descriptor, experiment_result)
  

#Run test - basically same as above, but only one model is run. 
def test_phase_experiments(phase_dict, letters):
    #first, get each parameter dicts for the letters in the experiment
    #There is one parameter dict for every letter in the  
    parameter_dict_list = create_letter_library.return_letter_parameter_dict_list(phase_dict, letters)
    #For each letter in our slate:
    for param_dict in parameter_dict_list:
        #Get all the dataset descriptors for that letter 
        dataset_descriptors_list = create_dataset_descriptors.create_dataset_descriptor_list_from_parameter_dict(param_dict)
        #For every dataset descriptor we generated: 
        for single_dataset_descriptor in dataset_descriptors_list:
            #Print the dataset descriptor
            print()
            print("Dataset Descriptor")
            print(json.dumps(single_dataset_descriptor, indent=4))
            #create a dataset result
            dataset_result = create_dataset_library.create_dataset_from_dataset_descriptor(single_dataset_descriptor)
            #Save the descriptor and result into the correct folder
            save_descriptor_and_result(single_dataset_descriptor, dataset_result)
            #For each scaling factor (experimental model modification) in a given dataset descriptor:
            for scaling_factor in single_dataset_descriptor["scaling_factors"]:
                print(scaling_factor)
                #We create that experiment descriptor 
                experiment_descriptor = create_experiment_descriptors.create_experiment_descriptor(scaling_factor, single_dataset_descriptor, dataset_result)
                #Print the experiment descriptor
                print()
                print("Experiment Descriptor")
                print(json.dumps(experiment_descriptor, indent=4))
                #These are auto-saved at this point. 
                single_dataset_descriptor, dataset_result, experiment_object, experiment_result = create_model.experiment_from_experiment_object(single_dataset_descriptor, dataset_result, experiment_descriptor)
                #Create the metrics 
                create_metrics.visualize_and_analyze(single_dataset_descriptor, dataset_result, experiment_descriptor, experiment_result)
                break
            break 



