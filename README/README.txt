## Data set

Data from 55 air quality monitoring stations, including PM2.5 concentration, temperature, air pressure, relative humidity and wind speed and direction, were collected over a 60-month period from January 2015 to December 2019, covering all cities in Shanxi Province.

## Experimental environment

+ python >= 3.6
+ tensorflow >= 1.15.0
+ numpy
+ pandas
+ datetime
+ PIL
+ Pillow
+ os

## This paper is experimentally weighted convlstm
1. image.py -- Draw an image. For example, to draw the image corresponding to PM2.5 on a certain Province.
2.data_helper.py-- loads the data and splits the training set and test set in a 7:3 ratio
3.Main. py --The main function of the model, which sets the parameters of the model
+ test_size: float, which sets the proportion of the test set
+ image_height: specifies the height of the image
+ image_width: specifies the width of the input image
+ image_in_channels: int, the number of channels to enter the image       
+ time_step: an integer indicating a move time step
+ num_stack: int, Convlstm number of stacks
+ batch_size: int
+ layers_id : list
+ confusion_type : 'mean' or 'sum'
+ conv_lstm_kernel: integer, convolution kernel
+ learning_rate: float, such as 0.0001
+ epochs: int, epoch to train
+ expand_method : 'manual' or 'auto'
4.model.py--The loaded data is brought into the model for training, evaluation, and saving predictions.
--To run the code, run the main function main.py directly

The code address of this paper is http://github.com/MICVRGroup/haze-code.
The code address of SPDE is https://github.com/guidofio ravanti/spde_spatio_temporal_pm10_modelling_italy.
The code address of D-Stem is https://code.google.com/p/d-stem/
The code for creating map can be found in image.py
The code for the ARIMA model can be found in ARIMA.txt
The code for the SPDE model can be found in SPDE.txt
The code for the D-Stem model can be found in D-Stem.txt
The code for the 3D-CNN model can be found in file 3DCNN
The code for the Conv-LSTM model can be found in convlstm.py
The code for the LSTM model can be found in lstm.py

 
