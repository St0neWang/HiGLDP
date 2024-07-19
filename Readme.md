# HiGLDP

## Overview


## Code

### `Dataset/matlab/` directory
- `RWR.m` RWR algorithm
- `GIP.m` gaussian interaction profile kernels similarity function
- `merge.m` the function to put the features of lncRNA and disease together respectively
- `run_merge.m` run the **merge.m**
- `similarity_generate.m` obtain 7 similarity networks based on the data in the `Dataset/data/` folder

### `Dataset/DenoisingAutoencoder/` directory
- `DAE.py` Denoising Autoencoder implementation
- `run_DAE.py` run `DAE.py` to obtain low dimensional feature

### `Dataset/` directory
- `data_concat.py` obtain the features of all lncRNA-disease association pairs
- `dataprocess.py` uses KNN to get feature graph

### `./` directory
- `config.py` the configure of model
- `model.py` HiGLDP model
- `function.py` some functions used in model
- `main.py` train model and predict

## data(我觉得可有可无)



## Useage

### dataprocess
- run `./Dataset/matlab/similarity_generate.m`
- run `./Dataset/matlab/run_joint.m`
- run `./Dataset/DAE/run_DAE.py`
- run `data_concat.py` to obtain the features of all lncRNA-disease pairs
- run `dataprocess.py` to get feature graph by KNN


### model training and predicted results
- run `main.py`

## Requirments

```python
networkx==3.1
numpy==1.23.5
pandas==1.5.3
scikit_learn==1.3.0
scipy==1.10.1
tensorflow==2.10.0
torch==1.13.1
torch_geometric==2.3.1
tqdm==4.66.1
```