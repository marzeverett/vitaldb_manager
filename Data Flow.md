# Data Flow
Purpose: To describe how data flows through this system

## High Level:

### Characterizing Experiments 
Phase: Phase describes the 'slate' of experiments being run. Different phases correspond to different experimental parameters such as model size/training type, prediction vs regression task, etc. Phase tries to encompass the most logical separations of high-level task questions. What a phase is doing/investigating is typically documented in its own code file. 

Letter: The Letter of an experiment describes a particular model/data combination/Network scheme. For instance, A describes Network 1, with data full together, and a LSTM model. B describes Network 1, datastreams separate, and an LSTM model. E describes a Network 2 pre-cursor, with data streams separate, and an Autoencoder model. 

#Model Type: Prediction, Regression, or Autoencoder Task.

Model Group: group index for model used under the different tasks (Identified by dictionary). Some tasks may use more than one model type (i.e. deep vs shallow lstm model, if applicable).

Datastream Index: an index corresponding to a whether a letter is using all datastreams together or using them separately. (0 together, 1 separate)

Location Index: an index corresponding to whether a letter is using all locations (cases) togther or using them separately (0 together, 1 separate). For these models, no separate locations are anticipated. (0 together, 1 separate). 

Input Index: The number of samples of input (in seconds, in this case). This is not expected to change across experiments. (30 seconds input anticipated for all)

Output Index: The number of samples of output (in seconds, in this case). This is not expected to change across experiments. (10 seconds output anticipated for all). 


### Assumptions for this Dataset:
- Data already downloaded as csv files (in vital_csvs folder) with caseid names
- Data already contains columns (features) we are working with
- Different cases variables share same feature names 
- Categorical variables already encoded 

### High Level Data Flow
Phase Descriptor -> Dataset Descriptor -> Experiment Descriptor -> Experimental Run -> Individual Metrics and Analysis

When Phase is finished, broad analysis can be completed. 

### High Level File Structure

Note: Need to fix the ability to reflect the particular model nodes as well as the model index (esp. in experiment)
Full Name of Dataset: {phase}_{letter}_{model_group}_{datastream_scheme}_{location_scheme}_{datastream_index}_{location_index}_{input_samples}_{output_samples}

Full name of given experiment path: experiments/{phase}_{letter}_{model_index}/{phase}_{letter}_{model_group}_{datastream_index}_{location_index}_{input_samples}_{output_samples}

Metric path for given experiment: metrics/{phase}_{letter}

dataset_generator - folder contains most of the code files for running phases

    generated files - folder contains files for experiments
        descriptors - contains basic dataset and experiment     descriptors
            (phase_letter)
        datasets - contains the dataset files (full_name)
        experiments - contains all specific experiment info (phase_letter) - sub folder (full_name)
        metrics - contains broad level excel metrics (phase_letter)

data_analysis
    main_metrics - folder contains files for metrics
    {phase_letter} - sub folders containing analysis for each experiment 

### To-Do: 

 - [ ] Create Dictionaries for Each Index Type
 - [ ] Better scheme for generating Dataset Descriptors 
 - [ ] Document more completely the different types of descriptors. 
 - [ ] Document metrics more completely (w/graph and visualize)
 - [ ] $match keyword which matches a parameter based on an existing parameter map 


## In the Code 
Phase File - contains high level information (not all needed for all experiments) about the experimental slate (phase_dict)
 
|

Letters File - contains info about how to generate descriptors from letters (add letters dict)

|

Dataset Descriptors File - contains info about how to generate dataset descriptors based on phase and letter information 

|

Experiment Descriptors File - contains info about how to generate experiment descriptos based on phase, letters, and dataset descriptors

|

Dataset Generator - generates datasets based on dataset descriptors

|

Model Generator - Generates and runs models based on an experiment descriptor and dataset desciptors

|

Graph and Analyze - analyzes and/or graphs info based on experiment results, adds to main metrics files. 

|

Test_Experiments - returns test experiments based on existing parameters. 


For ONE Letter, what can vary? 
Datasets 
Nope, we do -- multiple models can use the same dataset
datastream index can vary (not necessarily, though)
location index can vary (not in this dataset, though)
input samples
output samples can vary 

On experiment side - 
what model index it is can vary - do we even want to do this for this dataset??? -- Yeah probably. 




