# vitaldb_manager
AI experiment code on vitaldb dataset


# Purpose
For these experiments, we are looking at how the structure of a neural network on multivaraite, time sequence data might influence the accuracy of that network's predictions. 


# Data Streams
We are going to try out 5 input data streams from the dataset and will restrict the number of samples to those that contain these datastreams.

## Input Data Streams (Tables from Documentation)
### Demographic Information 

| Datastream label | Datastream Description | Type/Hz | Unit | 
| --- | --- | --- | | --- | 
| age | Age | N | years |
| sex |  Sex | N | M/F |
| height | Height | N | cm |
| weight | Weight | N | kg |
| bmi | Body mass index | N | kg/m2 | 
| asa | ASA Physical status classification | N | 1-5? | 

### Operation Information 
We are going to include: 

| Datastream label | Datastream Description | Type/Hz | Unit | 
| --- | --- | --- | | --- | 
| casestart | The recording start time | N | sec (0) |  
| caseend | The recording end time from casestart | N | sec | 
| anestart | The Anesthesia start time from casestart | N | sec |
| aneend | The Anesthesia end from from casestart | N | sec | 
| dis | Discharge time from casestart | N | sec |s 


----
| emop | Emergency operation | N | bool? | 
| optype | Surgery type | N | bool? | 
| los_postop | Postoperative length of hospital stay | N | days |
| preop_htn | Preoperative hypertension | N | bool? | 
| preop_dm | Preoperative diabetes | N | bools? | 

Prediction: 

| Datastream label | Datastream Description | Type/Hz | Unit | 
| --- | --- | --- | | --- | 
| death_inhosp |  In-hospital Mortality | N | bool? | 


### ECG Information from Tram-Rac 4A 

| Datastream label | Datastream Description | Type/Hz | Unit | 
| --- | --- | --- | | --- | 
| SNUADC/ART | Arterial pressure wave | W/500 | 




## Notes 
There may be an advantage in allowing small datastreams that aren't time-based to be incorporated without being time-autoencoded. 
Also may be an advantage in using a small preprocessing 1-D Conv Neural Network? Instead of the autoencoder. 