# BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks

This is a refactored implementation of BigST model as described in the following paper: [BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks, VLDB 2024].

### Requirements
The implementation requires python 3.10.  
All the packages could be installed via "pip intall <package_name>".  
```  
numpy
pandas
scipy
torch
```

### Hardware environment
Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz
NVIDIA A40 48GB

### Dataset
The California dataset is collected from the Caltrans Performance Measurement System (PeMS: https://pems.dot.ca.gov/). We will make the processed datasets used in experiments available soon.

### Usage
If you want to conduct the model training process, you can use the following command.

```
python run_model.py
```

