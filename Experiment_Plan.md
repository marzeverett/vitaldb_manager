# Experimental Plan 

15 Letters (So Far) - Might take out (1-3 of them?) if decide not to do ensemble on case 

## Base (Network 1)
A: 
Run LSTM for all datastreams, using all cases. However, must generalize - can't differentiate as with the the locations, since time observations are not synced. 
Network 1, all datastreams and all cases 

B: 
All datastreams from one case used to predict all datastreams
from one case 

C: 
One datastream from all cases used to predict one datastream from all cases 

D: (No) Think we are not going to do this since it isn't feasible for the amount of data and cases we have. 


## AE Separate Latent Space (Network 2)
### By Datastream 
E. One datastream from all cases used to recreated one datastream from all cases, using an AE 

F. One datastream from all cases used to predict one datastream from all cases, using LSTM network with data preprocessed by E 

G. ALl datastreams from all cases predict all datastreams from all locations fusing data preprocessed by E 

H. (No)

I. (No)

J. (No)

### By Case  - Check 

L: All datastreams from one case used to recreate all datastreams from one case, using an AE 

M. All datastreams from one case used to predict all datastreams from one case, using an LSTM network with data preprocessed by autoencoders from L.

N.(NO) All datastreams from all cases used to predict all datastreams from all cases using an LSTM network with data preprocessed by autoencoders from L and fused for each case (Ensemble?)

Q. (No) 

## AE Separate Latent Space, then Feature Merged Latent Space (Network 3)

## By datastream 
S. All datastreams from all cases recreated from all datastreams from all locations, after being preprocessed and fusing input from aes in E (on datastream)

T. All datastreams from all cases predicted from all datastreams from all cases, after being preprocessed by AE in S

U. (No)

V. (No)

W. (No)

## By Case - Revisit 

X. All datastreams from all cases are recreated from all datastreams from all cases, preprocessing and fusing input from L (on case) -- Ensemble?  (No)

Y. All datastreams from all cases are predicted from all datastreams from all cases, after being preprocessing by AE in X.  (No)

Z. (No)

AA. (No)

AB. (No)

## AE datastreams/locations share Latent Space

AC. All datastreams from all cases are recreated from all datastreams from all cases

AD. All datastreams from all cases are predicted from all datastreams from all cases, after being preprocessed by the AE from AC. 





Prediction Letters - A, B, C, F, G, M, T, AD



