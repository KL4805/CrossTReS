# CrossTReS
This is the repo for paper "Selective Cross-City Transfer Learning for Traffic Prediction via Source City Region Re-Weighting", KDD 2022. 

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
- `gen_rt_dict.py`: This script generates the dictionary for RegionTrans to do matching. 

You can check the tunable parameters in `run_crosstres.py` and `run_crosstres_rt.py`. 

Note: Running`run_crosstres.py` requires approximately 10GB GPU memory with batch_size=32. You can reduce batch_size to reduce memory cost. 

## Procedures to run the scripts
- `run_crosstres.py`: `python run_crosstres.py --SET_PARAMETERS`. 
- `run_crosstres_rt.py`: 
    - First, run `python gen_rt_dict.py --metric poi --source [NY, CHI] --target [DC]`. You will get a file under the `src/rt_dict` folder. 
    - Then, run `python run_crosstres_rt.py --SET_PARAMETERS --rt_dict poi`. 
