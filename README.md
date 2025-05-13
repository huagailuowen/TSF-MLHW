# Time Series Forecasting

## Overview

Use some traditional time series forecasting methods together with crossformermodel to predict the future values of a time series. The possible research methods include:
> 1. use traditional algorithms to generate extra features, and then use crossformer to predict the future values of a time series. This may accelerate the convergence of the model and improve the generalization ability of the model, given the limited data size.
(use sin to represent the periodicity of the time series, use ARIMA to represent the trend of the time series, use Holt-Winters to represent the seasonality of the time series, etc.)

> 2. use traditional algorithms to create a rough prediction, and then use crossformer to refine the prediction by predicting the residuals.

> 3. use different algorithms and ensemble the results to improve the accuracy of the prediction, it can be a route or just a simple average.

> 4. try to enhance the efficiency of the crossformer by using sparce attention or linear attention.


## Requirements
