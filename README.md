# Time Series Forecasting

## Overview

Use some traditional time series forecasting methods together with crossformermodel to predict the future values of a time series. The possible research methods include:
> 1. use traditional algorithms to generate extra features, and then use crossformer to predict the future values of a time series. This may accelerate the convergence of the model and improve the generalization ability of the model, given the limited data size.
(use sin to represent the periodicity of the time series, use ARIMA to represent the trend of the time series, use Holt-Winters to represent the seasonality of the time series, etc.)
given that the crossformer is better at analyzing the relationship between different time series, we can use the crossformer to predict the future values of a time series with the help of other time series.

> 2. use traditional algorithms to create a rough prediction, and then use crossformer to refine the prediction by predicting the residuals.

> 3. use different algorithms and ensemble the results to improve the accuracy of the prediction, it can be a route or just a simple average.

> 4. try to enhance the efficiency of the crossformer by using sparce attention or linear attention.

## Method

1. use drift windows and seasonal decomposition to extract the trend and seasonality of the time series, and use average to get the mean of the overlap part of the relevant windows to generate the final season prediction feature.
with window size 3 times of the period of the time series might be a good choice.

2. use the sin to represent the periodicity of the time series, we can use both weekly and daily periodicity to generate the final periodicity prediction feature.

3. a problem, this drift window method would cause the trend not smooth enough. 
## Requirements


## Test
```bash
zsh scripts/ETTh1-f.sh
python eval_crossformer.py --checkpoint_root ./checkpoints --setting_name Crossformer_ETTh1-f_il720_ol168_sl24_win2_fa10_dm256_nh4_el3_itr0
```

## Results

1. train_original_only = false, --in_len 720 --out_len 168 --seg_len 24 --itr 1
mse:0.3881882429122925, mae:0.41910073161125183

2. train_original_only = true, --in_len 720 --out_len 168 --seg_len 24 --itr 1
mse:0.4016232192516327, mae:0.43583056330680847
(may induce the overfitting of the model or the extra feature can help the model to learn the trend and seasonality of the time series better)


