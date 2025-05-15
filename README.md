# Time Series Forecasting

## Overview

Use some traditional time series forecasting methods together with crossformermodel to predict the future values of a time series. The possible research methods include:
> 1. use traditional algorithms to generate extra features, and then use crossformer to predict the future values of a time series. This may accelerate the convergence of the model and improve the generalization ability of the model, given the limited data size.
(use sin to represent the periodicity of the time series, use ARIMA to represent the trend of the time series, use Holt-Winters to represent the seasonality of the time series, etc.)

> 2. use traditional algorithms to create a rough prediction, and then use crossformer to refine the prediction by predicting the residuals.

> 3. use different algorithms and ensemble the results to improve the accuracy of the prediction, it can be a route or just a simple average.

> 4. try to enhance the efficiency of the crossformer by using sparce attention or linear attention.

## Method

1. use drift windows and seasonal decomposition to extract the trend and seasonality of the time series, and use average to get the mean of the overlap part of the relevant windows to generate the final season prediction feature.
with window size 3 times of the period of the time series might be a good choice.

2. use the sin to represent the periodicity of the time series, we can use both weekly and daily periodicity to generate the final periodicity prediction feature.

3. a problem, this drift window method would cause the trend not smooth enough. 
## Requirements
