# BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks

This is a refactored implementation of BigST model as described in the following paper: [BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks, VLDB 2024].

## Requirements
`python3`, `numpy`, `pandas`, `scipy`, `torch`

## Hardware environment
Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz
NVIDIA A40 48GB

## Datasets
The California dataset is downloaded from the Caltrans Performance Measurement System ([PeMS](https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit)) website, we use Station 5-Minute traffic speed data ranging from 2022-04-01 to 2022-06-31.

## Usage
To preprocess the long historical traffic time series:

```
python preprocess/preprocess.py
```

If you want to conduct the model training and prediction process, you can use the following command:

```
python run_model.py
```

## Citation
If you find our work is useful for your research, please consider citing:

```
@article{hanbigst,
  title={BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks},
  author={Han, Jindong and Zhang, Weijia and Liu, Hao and Tao, Tao and Tan, Naiqiang and Xiong, Hui},
  journal={Proceedings of the VLDB Endowment},
  volume={17},
  number={5},
  pages={1081--1090},
  year={2024},
  publisher={VLDB Endowment}
}
```
