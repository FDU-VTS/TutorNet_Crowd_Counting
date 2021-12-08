# TutorNet_Crowd_Counting

This repository contains codes for the official implementation in PyTorch of paper ["Learning Error-Driven Curriculum for Crowd Counting"](https://arxiv.org/abs/2007.09676).


## Main Results of Our Method

| Method | MAE | MSE | 
| --------   | -----:   | :----: |
|MCNN |26.4 |41.3|
|MCNN+SF| 15.3 |35.2|
|MCNN+SF+TN| 14.4 |25.1|


| Method | MAE | MSE | 
| --------   | -----:   | :----: |
|CSRNet| 10.6 |16.0|
|CSRNet+SF| 10.4 |15.9|
|CSRNet+SF+TN| 9.4 |15.6|


| Method | MAE | MSE | 
| --------   | -----:   | :----: |
|U-net| 26.8| 39.7|
|U-net+SF| 13.5| 23.0|
|U-net+SF+TN| 12.1 |19.7|


| Method | MAE | MSE | 
| --------   | -----:   | :----: |
|DenseNet| 13.0| 22.7|
|DenseNet+SF| 7.5| 12.8|
|DenseNet+SF+TN| 7.0| 12.2|



## Train
1. Download the data and use the script in /pre_process.

2. python train_DenseNet_shanghai.py

***If you have any questions, do not afraid to contact us.***