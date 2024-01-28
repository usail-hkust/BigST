# BigST

This is a pytorch implementation of BigST model as described in the following paper: [BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks].

### Create conda environment and install packages
The implementation requires python 3.10.  
```
conda create -n bigst python==3.10
conda activate bigst
``` 

All the packages could be installed via "pip intall <package_name>".  
```  
numpy==1.22.3
pandas==1.5.3
pytorch-lightning==2.0.4
scikit-base==0.6.1
scikit-learn==1.1.1
scipy==1.8.0
torch==2.0.1
torch-cluster==1.6.1+pt113cu116
torch-geometric==2.3.0
torch-scatter==2.1.1+pt113cu116
torch-sparse==0.6.17+pt113cu116
torch-spline-conv==1.2.2+pt113cu116
torchaudio==0.13.1+cu116
torchdiffeq==0.2.3
torchmetrics==0.11.4
torchvision==0.15.2
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

