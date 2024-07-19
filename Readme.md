# HiGLDP

## Overview
A computational model designed to predict lncRNA-disease associations through multi-omic integration and advanced graph neural network techniques.

## Code

### `Dataset/matlab/` directory
- `RWR.m` RWR algorithm
- `GIP.m` gaussian interaction profile kernels similarity function
- `merge.m` the function to put the features of lncRNA and disease together respectively
- `run_merge.m` run the **merge.m**
- `similarity_generate.m` obtain 7 similarity networks based on the data in the `Dataset/data/` folder

### `Dataset/DenoisingAutoencoder/` directory
- `DAE.py` Denoising Autoencoder implementation
- `runDAE.py` run `DAE.py` feature extraction

### `Dataset/` directory
- `data_concat.py` obtain the features of all lncRNA-disease association pairs
- `dataprocess.py` uses cosine similarity to get association feature graph

### `./` directory
- `config.py` the configure of model
- `model.py` HiGLDP model
- `function.py` some functions used in model
- `main.py` train model and test


## Data

### `Dataset/data/` directory
- `lncid.txt` lncRNA id
- `disid.txt` disease id
- `mat_DO_circRNA.txt` disease-circRNA association matrix
- `mat_DO_Metabolite.txt` disease-metabolite association matrix
- `Similarity_Matrix_DO.txt` disease similarity from FNSemSim method
- `mat_Lnc_DO.txt` lncRNA-disease association matrix
- `mat_Lnc_mRNA.txt` lncRNA-mRNA association matrix
- `mat_Lnc_Protein.txt` lncRNA-protein association matrix
- `mat_Lnc_RBP.txt` lncRNA-RBP association matrix
- `index.txt` all indices of lncrna-disease pairs for training and testing
- `label.txt` the labels for index file **index.txt**

### `5fold` and `10fold` directory
the datasets for 5-fold and 10-fold cross validation
- `edgex.txt` the interconnected graph of train dataset
- `trainx.txt` the lncrna-disease pairs' indices for training
- `testx.txt` the lncrna-disease pairs' indices for testing




## Useage

### dataprocess
- run `./Dataset/matlab/similarity_generate.m`
- run `./Dataset/matlab/run_joint.m`
- run `./Dataset/DenoisingAutoencoder/runDAE.py`
- run `./Dataset/data_concat.py`
- run `./Dataset/dataprocess.py`


### model training and test results
- run `main.py`


