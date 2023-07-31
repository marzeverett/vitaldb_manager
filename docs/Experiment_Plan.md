# Experimental Plan 

12 Letters

## Base (Network 1)
A: 
Run LSTM for all datastreams, using all cases. However, must generalize - can't differentiate as with the the locations, since time observations are not synced. 
Network 1, all datastreams and all cases 

B: 
All datastreams from one case used to predict all datastreams
from one case 

C: 
One datastream from all cases used to predict one datastream from all cases 


## AE Separate Latent Space (Network 2)
### By Datastream 
E. One datastream from all cases used to recreated one datastream from all cases, using an AE 

F. One datastream from all cases used to predict one datastream from all cases, using LSTM network with data preprocessed by E 

G. All datastreams from all cases predict all datastreams from all locations fusing data preprocessed by E 


### By Case  - Check 

L: All datastreams from one case used to recreate all datastreams from one case, using an AE 

M. All datastreams from one case used to predict all datastreams from one case, using an LSTM network with data preprocessed by autoencoders from L.


## AE Separate Latent Space, then Feature Merged Latent Space (Network 3)

## By datastream 
S. All datastreams from all cases recreated from all datastreams from all locations, after being preprocessed and fusing input from aes in E (on datastream)

T. All datastreams from all cases predicted from all datastreams from all cases, after being preprocessed by AE in S


## AE datastreams/locations share Latent Space

AC. All datastreams from all cases are recreated from all datastreams from all cases

AD. All datastreams from all cases are predicted from all datastreams from all cases, after being preprocessed by the AE from AC. 





Prediction Letters - A, B, C, F, G, M, T, AD

AEs: E, L, S, AC 

## Plan
1. Run all with shallow models
2. Run all with deeper models
3. Run with Retraining
4. Run with Prediction 


Make sure to discuss limitations 

Notes: We also might want to try all per one case or one per one case, but not introduce the additional post-processing. Just as a sanity check since a person is likely a more complex biological system than what we have for weather. 