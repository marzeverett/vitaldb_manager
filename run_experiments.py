import json 
import create_letter_library
import create_dataset_descriptors
import create_dataset_library 
import create_experiment_descriptors 
import create_model 
import create_metrics 
# generated_files/descriptors/phase_letter/{files}
# generated_files/datasets/phase_letter/dataset_name/{files}
# generated_files/experiments/phase_letter/model/dataset_name/{files}
#generated_files/experiments/metrics/
#Going to have to figure out a test here. 

def run_phase_experiments(phase_dict, letters):
    #first, get each parameter dicts for the letters in the experiment 
    parameter_dict_list = create_letter_library.return_letter_parameter_dict_list(phase_dict, letters)
    #For each letter, get the dataset descriptors
    for param_dict in parameter_dict_list:
        #Get all the dataset descriptors for each letter
        dataset_descriptors_list = create_dataset_descriptors.create_dataset_descriptor_list_from_parameter_dict(param_dict)

        #For single_dataset_descriptor in the descriptors for a letter 
        dataset_result = create_dataset_library.create_dataset_from_dataset_descriptor(dataset_descriptors_list[0])
        
        for scaling_factor in dataset_descriptors_list[0]["scaling_factors"]:
            experiment_descriptor = create_experiment_descriptors.create_experiment_descriptor(scaling_factor, dataset_descriptors_list[0], dataset_result)
            #print(json.dumps(experiment_descriptor, indent=4))
            
            #These are auto-saved at this point. 
            dataset_descriptor, dataset_result, experiment_object, experiment_result = create_model.experiment_from_experiment_object(dataset_descriptors_list[0], dataset_result, experiment_descriptor)
            #Create the metrics 
            create_metrics.visualize_and_analyze(dataset_descriptor, dataset_result, experiment_descriptor, experiment_result)
            create_metrics.just_visualize(dataset_descriptor, dataset_result, experiment_descriptor, experiment_result)
            break

        #Here you would save the the dataset descriptors and dataset result 
        #Inside a for loop 
        #For each dataset descriptor, going to generate (several) experiment descriptors
        break
        #Now - we actually make the datasets. Which is going to be somewhat trickier. 




