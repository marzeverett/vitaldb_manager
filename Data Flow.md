# Data Flow
Purpose: To describe how data flows through this system

## High Level:

### Characterizing Experiments 
Phase: Phase describes the 'slate' of experiments being run. Different phases correspond to different experimental parameters such as model size/training type, prediction vs regression task, etc. Phase tries to encompass the most logical separations of high-level task questions. What a phase is doing/investigating is typically documented in its own code file. 

Letter: The Letter of an experiment describes a particular model/data combination/Network scheme. For instance, A describes Network 1, with data full together, and a LSTM model. B describes Network 1, datastreams separate, and an LSTM model. E describes a Network 2 pre-cursor, with data streams separate, and an Autoencoder model. 

Model Type: Prediction, Regression, or Autoencoder Task.

Model Index: Index for model used under the different tasks (Identified by dictionary). Some tasks may use more than one model type (i.e. deep vs shallow lstm model, if applicable).

Datastream Index: an index corresponding to a whether a letter is using all datastreams together or using them separately. (0 together, 1 separate)

Location Index: an index corresponding to whether a letter is using all locations (cases) togther or using them separately (0 together, 1 separate). For these models, no separate locations are anticipated. (0 together, 1 separate). 

Input Index: The number of samples of input (in seconds, in this case). This is not expected to change across experiments. (30 seconds input anticipated for all)

Output Index: The number of samples of output (in seconds, in this case). This is not expected to change across experiments. (10 seconds output anticipated for all). 


### To-Do: 

 - [ ] Create Dictionaries for Each Index Type
 - [ ] Better scheme for generating Dataset Descriptors 
 - [ ] Document more completely the different types of descriptors. 
 - [ ] Document metrics more completely (w/graph and visualize)
     


## In the Code 


