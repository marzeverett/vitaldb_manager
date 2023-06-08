# vitaldb_manager
AI experiment code on vitaldb dataset


# Purpose
For these experiments, we are looking at how the structure of a neural network on multivaraite, time sequence data might influence the accuracy of that network's predictions. 


# Data Streams
We are going to try out 5 input data streams from the dataset and will restrict the number of samples to those that contain these datastreams.

## Input Data Streams (Tables from Documentation)

### Demographic Information (Non Time Sequence)

| Datastream label | Datastream Description | Type/Hz | Unit | 
| --- | --- | --- | --- | 
| age | Age | N | years |
| sex |  Sex | N | M/F |
| height | Height | N | cm |
| weight | Weight | N | kg |
| bmi | Body mass index | N | kg/m2 | 
| asa | ASA Physical status classification | N | 1-5? | 

### Operation Information (Non Time Sequence)
We are going to include: 

| Datastream label | Datastream Description | Type/Hz | Unit | 
| --- | --- | --- | --- | 
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
| --- | --- | --- | --- | 
| death_inhosp |  In-hospital Mortality | N | bool? | 


### ECG Information from Tram-Rac 4A 

| Datastream label | Datastream Description | Type/Hz | Unit | 
| --- | --- | --- | --- | 
| SNUADC/ART | Arterial pressure wave | W/500 | mmHg | 
| SNUADC/CVP | Central venous pressure wave | W/500 | mmHg | 
| SNUADC/ECG_II | ECG lead II wave | W/500 | mV | 
| SNUADC/ECG_V5 | ECG lead V5 wave | W/500 | mV | 
| SNUADC/FEM | Femoral arterial pressure wave | W/500 | mV | 

### Ventilator Information from Solar8000

| Datastream label | Datastream Description | Type/Hz | Unit | 
| --- | --- | --- | --- | 
| Solar8000/VENT_INSP_TM | Inspiratory time (from ventilator) | N | sec | 
| Solar8000/VENT_MAWP | Mean airway pressure (from ventilator) | N | mbar | 
| Solar8000/VENT_MV | Minute ventilation (from ventilator) | N | L/min | 
| Solar8000/VENT_PIP | Peak inspiratory pressure (from ventilator) | N | mbar | 
| Solar8000/VENT_PPLAT | Plateau pressure (from ventilator) | N | mbar | 
| Solar8000/VENT_RR | Respiratory rate (from ventilator) | N | /min | 
| Solar8000/VENT_TV | Measured tidal volume (from ventilator) | N | mL | 
| Solar8000/BT | Body temperature | N | degrees C | 

### Anesthesia Information from Orchestra 

| Datastream label | Datastream Description | Type/Hz | Unit | 
| --- | --- | --- | --- | 
| Orchestra/RFTN20_CE | Effect-site concentration (remifentanil 20 mcg/mL) | N | ng/mL |
| Orchestra/RFTN20_CP | Plasma concentration (remifentanil 20 mcg/mL) | N | ng/mL |
| Orchestra/RFTN20_CT | Target concentration (remifentanil 20 mcg/mL) | N | ng/mL |
| Orchestra/RFTN20_RATE | Infusion rate (remifentanil 20 mcg/mL) | N | mL/hr |
| Orchestra/RFTN20_VOL | Infused volume (remifentanil 20 mcg/mL) | N | mL |

## ECG Information from BIS

| Datastream label | Datastream Description | Type/Hz | Unit | 
| --- | --- | --- | --- | 
| BIS/BIS | Bispectral index value | N | unitless |
| BIS/EEG1_WAV | EEG wave from channel 1 | W/128 | uV |
| BIS/EEG2_WAV | EEG wave from channel 2 | W/128 | uV |
| BIS/EMG | Electromyography power | N | dB |
| BIS/SEF | Spectral edge frequency | N | Hz |
| BIS/SQI | Spectral quality index | N | % |
| BIS/SR | Supression Ratio | N | % |
| BIS/TOTPOW | Total power | N | dB |


## Notes 
There may be an advantage in allowing small datastreams that aren't time-based to be incorporated without being time-autoencoded. (Ensemble Learn)
Also may be an advantage in using a small preprocessing 1-D Conv Neural Network? Instead of the autoencoder. 