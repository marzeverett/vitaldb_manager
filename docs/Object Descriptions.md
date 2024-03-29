# Object Descriptions

## Phase Dict 

    {
        "phase_name": "1",
        "phase_path_start": "generated_files/",
        "delete_stream": False,
        "task_type": "regression",
        "model_group": "0",
        "input_samples": [30],
        "output_samples": [10],
        "test": False,
        "retrain": False,
        "predict_type": None,
        "building_on_phase": False,
        "retrain_dict": {
        "retrain_from_phase": "4",
            "separation_scheme": "ds",
            "retrained_letters": ["E", "H"],
        }
    }



## Parameter Dict 

## Dataset Descriptor
A Dataset a dictionary with the following keys:
"datasets": values is a list of datasets you want to use for input (in this case, case names)

"input_fields": a list of input fields

"output_fields": a list of output fields 

"input_samples": Number of samples (days, in this case) to put into a single input sequence

"output_samples": Number of samples (days, in this case) to put into a single output sequence

"output_slices_offset": Days of offset from the edge of an input sequence to an output sequence (in most cases, just 1)

"task_type": type os ML task, either regression or prediction 
 
"dataset_name": The name to give the dataset 

#Added 
"normalization_data": A dictionary keyed by field, corresponding to dict of with max and min values keys. 

"dataset_folder_path": the parent folder of the dataset

"dataset_class": A dictionary, which could have the following keys:
    "location_scheme": the index the location level scheme (0-4) being used 
    "datastream_scheme" the index of the data stream level scheme (0-4) being used
    "l_combo": the index of the combination of input/output locations being used
    "ds_combo": the index of the combination of input/output data streams being used 
    "input_days": the number of input days
    "output_days": the number of output days 
    "version": version of the experiments being run  

"x_columns": list of names, in order, of the x columns.

"y_columns": list of names, in order, of the y columns.

"test": Bool, whether or not a test run 

"target_model": "ae" or "time_regression" or "predict" depending on the target model for the dataset 

"delete_stream" - name or list of names of datastreams to delete 

"phase_metrics" - name of metrics file

"base_dataset_name": phase and letter identifying base dataset

"base_name": phase + letter + exp

"deep_lstm" T/F

"deep_ae": T/F 

#Don't think the below needed? 

"ae_synthesis" - optional, based on what we fuse one

"conv": T/F

"conv_and_prev_ae": T/F 

## Dataset Result

x_vect - 

y_vect - 

x_key - 

y_key - 

x_raw - 

y_raw - 

#Maybe x_normalized, y_normalized, x_unnormalized, y_unnormalized 


## Save paths

generated_files/descriptors/phase_letter/{files}
generated_files/datasets/phase_letter/dataset_name/{files}
generated_files/experiments/phase_letter_model/dataset_name/{files}



## Experiment Descriptor


## Experiment Result 