import json 


phase_dict = {
    "phase_name" = "1",
    "phase_path_start" = "generated_files/",
    "letters" = ['A', 'B', 'C', 'E', 'F', 'G', 'L', 
    'M', 'S', 'T', 'AC', 'AD'],
    "delete_stream" = False,
    "task_type" = "regression",
    "model_index" = "0",
    "input_samples" = [30],
    "output_samples" = [10],
    "test" = False
    "retrain" = False,
    "retrain_dict" = {
        "retrain_from_phase": "4",
        "separation_scheme": "ds",
        "retrained_letters": ["E", "H"],
    }
}

#ae model is prev base name concat with scaling factor 
#ae prev name is prev dataset name 

slate_library.run(phase_name, phase_path_start, letters, input_days, output_days, use_scaling_factor, prev_phase_base=prev_phase_base, test=test, transfer_learn=transfer_learn, transfer_dict=transfer_dict)

#IMPORTANT: How to handle the issue of pretraining? 
