# CrossTReS
Repo for CrossTReS: Cross-city Transfer Learning for Traffic Prediction via Source Region Selection

## Step 1: Data
Go to `data` repo and unzip the `crosstres_data.zip` file. 

## Step 2: Run the scripts in `src`
The structures of `src` are as follows: 
- `model.py`: Contains implementation of base models. 
- `utils.py`: Necessary utility functions. 
- `run_crosstres.py`: The implementation of CrossTReS. The requirements are: 
  -  Python=3.8 
  -  PyTorch=1.9.0
  -  DGL=0.6.1
  -  sklearn
- `run_crosstres_rt.py`: The implementation of CrossTReS which uses RegionTrans for fine-tuning. 

You can check the tunable parameters in `run_crosstres.py` and `run_crosstres_rt.py`. 

Note: Running`run_crosstres.py` requires approximately 10GB GPU memory with batch_size=32. You can reduce batch_size to reduce memory cost. 
