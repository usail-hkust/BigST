# BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks

This is a refactored implementation of BigST model as described in the following paper: [BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks, VLDB 2024].

## Requirements
`python3`, `numpy`, `pandas`, `scipy`, `torch`

## Hardware environment
Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz
NVIDIA A40 48GB

## Usage
To preprocess the long historical traffic time series:

```
python preprocess/preprocess.py
```

If you want to conduct the model training and prediction process, you can use the following command:

```
python run_model.py
```

