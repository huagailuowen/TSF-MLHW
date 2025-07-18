# Time Series Forecasting (fulfill the potential of crossformer with feature engineering)


## Introduction

crossformer is a transformer-based model for time series forecasting. It is designed to capture the temporal dependencies and relationships between different time series. The model uses a cross-attention mechanism to learn the interactions between different time series, allowing it to make accurate predictions based on the information from multiple sources, which is a great advantage over traditional time series forecasting methods.

However, the crossformer model may not be able to fully utilize its potential in some cases, especially when the data is limited or noisy. The strength of the crossformer model lies in its ability to learn complex relationships between different time series, but this requires a sufficient amount of feature and data to train on. In cases where the data is limited or noisy, the model may struggle to learn these relationships effectively, leading to suboptimal performance.

In this case, we can use some traditional time series forecasting methods to extract useful features from the data and improve the performance of the crossformer model. By combining the strengths of both traditional methods and crossformer, we can achieve better results in time series forecasting tasks.


## Overview

Use some traditional time series forecasting methods together with crossformermodel to predict the future values of a time series. The possible research methods include:
> 1. use traditional algorithms to generate extra features, and then use crossformer to predict the future values of a time series. This may accelerate the convergence of the model and improve the generalization ability of the model, given the limited data size.

Given that the crossformer is better at analyzing the relationship between different time series, we can use the crossformer to predict the future values of a time series with the help of other time series.

(use sin to represent the periodicity of the time series, use ARIMA to represent the trend of the time series, use Holt-Winters to represent the seasonality of the time series, etc. But after experiment we found that these 2 methods perform poorly in the ETTh1 dataset which is not smooth enough and has many kinds of periodical patterns and noise.)



> 2. use traditional algorithms to create a rough prediction, and then use crossformer to refine the prediction by predicting the residuals.

> 3. use different algorithms and ensemble the results to improve the accuracy of the prediction, it can be a route or just a simple average.

> 4. try to enhance the efficiency of the crossformer by using sparce attention or linear attention.

## Method

1. First we try to figure out the suitable methods to extract useful features from the time series. 

[image] : diagram/original_features.png

We analyze the time series and find that the time series has a strong and complex seasonality, and we can use some traditional methods to extract the trend and seasonality of the time series. Obviously we can't just simply use the stl decomposition to extract the trend and seasonality of the whole time series, for this would omit the local periodicity feature of the time series.

So we combine the stl decomposition with the drift windows method to extract the trend and seasonality of the time series by sampling out some windows from the time series, and then use stl decomposition for the data in the each window.

use drift windows and seasonal decomposition to extract the trend and seasonality of the time series, and use average to get the mean of the overlap part of the relevant windows to generate the final season prediction feature.

then we need to figure out the period and the window size we take.

We analyze the data of the first months and try out a lot of experiments.

We compare 2 decomposition methods, the stl decomposition and the seasonal decomposition, and we find that the stl decomposition is more precise than the seasonal decomposition, with a seasonal behavior that record more detail of the time series. So we use the stl decomposition to extract the trend and seasonality of the time series.

The major 2 choice of period is 24 and 168, which is the daily periodicity and weekly periodicity of the time series. 
However, we find that the weekly periodicity is not stable enough, when the window size is limited(the maximum size can't exceed the size of input data, which is 720). And the decomposition of the weekly periodicity is not precise enough for everyday as the dataset is a high-frequency dataset, where the data points are collected at short intervals (every 1 hour). And it is also too easily to become overfitted to the data, which is not good for the model to learn the periodicity of the time series.

[image] : diagram/composition_method/HUFL_stl_p167.png
[image] : diagram/composition_method/HUFL_stl_p23.png

So we decide to use the daily periodicity to extract the trend and seasonality of the time series, let the model to learn the weekly periodicity by itself(the trend feature we collected has already contain the weekly periodicity, and the model can learn the weekly periodicity by itself easily).


A window size with window size 7 times of the period of the time series might be a good choice, because we ensure both the stability of the decomposition and excavation of the local periodicity feature for different parts of the time series.

Furthermore, use 2-step seasonal decomposition with different period is also a promising method to extract the periodicity of the time series. First step we use the daily periodicity to extract the daily periodicity of the time series, and then use the weekly periodicity to extract the more subtle periodicity of the time series. This method can be used to extract the periodicity of the time series with different period, and then use the average to get the mean of the overlap part of the relevant windows to generate the final periodicity prediction feature.
[image] : diagram/LUFL_decomp_recon_p24.png
[image] : diagram/LUFL_special/LUFL_two_step_decomposition.png
We just use the above method in the feature LUFL, as we find out that the residual of the normal seasonal decomposition has periodicity.


2. use the sin to represent the periodicity of the time series, we can use both weekly and daily periodicity to generate the final periodicity prediction feature.


3. There is a problem, this drift window method would cause the trend not smooth enough. We take out some experiments, and find out savgol filter is a good choice to smooth the trend.
We use savgol filter to smooth the trend, and then use the smoothed trend to generate the final trend prediction feature. This would prevent the crossformer being overfitting to the origin trend with noise.
[image] : diagram/HUFL_decomposition_reconstruction_p24.png
[image] : diagram/smooth_method/rmse_overall_comparison.png

[image] : diagram/smooth_method/HUFL_decomp_recon_p24_savgol.png
[image] : diagram/smooth_method/HUFL_trend_smoothing_comparison_savgol.png

4. for the efficiency of the training process, we preprocess the data and save the features to a file, and then use the preprocessed data to train the model. But the drift window method would leak the future information to the past, so we need to use a mask to prevent the model from using the future information during the evaluation process by generating the extra feature online. This would prevent the model from using the future information during the evaluation process.

[image] : self draw an example of the drift window method

We have running several experiments to test if the training data with some extent of future information leakage(the output length is much longer than the size of window) would result in crash in evaluation process, and we found that the model can still work well in the evaluation process, showing strong generalization ability of the model.

5. As we produce the high quality extra feature, we suppose that the model can also learn from this artificial feature, which may teach the model to learn the trend and seasonality of the time series better.
So we carry out the experiment to train the model with the extra feature and the original feature together by changing the loss function to fit both the extra feature and the original feature when the parameter train_original_only is set to false, otherwise the model will only try to predict those original features.
We control the remaining parameters to test which way is better. 

## Requirements


## Test
```bash
cd Model
python cross_exp/precompute_causal_features.py --data_path datasets/ETTh1.csv --output_dir ./datasets/


zsh scripts/ETTh1-f.sh

python eval_crossformer.py --checkpoint_root ./checkpoints --setting_name Crossformer_ETTh1-f_il720_ol168_sl24_win2_fa10_dm256_nh4_el3_itr0
```

## Results

1. train_original_only = false, --in_len 720 --out_len 168 --seg_len 24 --itr 5(average the 5 iterations)
mse:0.38818824291, mae:0.4191007316112

2. train_original_only = true, --in_len 720 --out_len 168 --seg_len 24 --itr 5(average the 5 iterations)
mse:0.40162321925, mae:0.4358305633068
(may induce the overfitting of the model or the extra feature can help the model to learn the trend and seasonality of the time series better.
We know that the crossformer model is very sensitive to the input data, noise and lack of data can cause the model to overfit to the input data, 
leading to poor performance in the evaluation process.
In our hybrid training process, we push the model to learn the trend and seasonality of the time series by using the extra feature, not only focus on the original features. This may be a good way to improve the performance of the model.)

3. train_original_only = false, --in_len 720 --out_len 720 --seg_len 24 --itr 1
mse:0.4677826117, mae:0.4860403665

4. train_original_only = true, --in_len 720 --out_len 720 --seg_len 24 --itr 1
mse:0.5391083359, mae:0.5349105000 (poorer than the original model)
(this may show that only add the extra feature to the model is not enough, for we only focus on the forcasting preformance of the original features, adding features could interfere the cross attention layer. But if we also demand the model to learn the trend and seasonality of the time series, the performance of the model would be better than the original model, meaning that the traditional methods can help the model to learn the trend and seasonality of the time series better.)


5. Moreover, we find that using our method, the variance of the 5 iterations is much smaller than the original model, which means that our method can help the training process to be more stable and robust.

6. We also find that the model can still work well in the evaluation process which use the limited data with no future information to generate the extra feature, showing strong generalization ability of the model.
The performance only droped by about 0.002 in average in the real evaluation process, which is acceptable. Our method not only consider the training efficiency by use the average of the overlap part of the sample windows as the extra feature, but also ensure the prediction quality with no future information leakage in the evaluation process.

raw data(Results on the 720 input and 720 output length):

test_loss(with a little future information leakage): 
```
mse:0.4634285271167755, mae:0.48510023951530457
mse:0.46192309260368347, mae:0.48311248421669006
mse:0.46087634563446045, mae:0.4816077649593353
mse:0.47134190797805786, mae:0.48828035593032837
mse:0.46802905201911926, mae:0.4854092299938202

```
test_loss(with no future information leakage):
```
mse:0.4713256697654724, mae:0.48986250162124634
mse:0.46323874592781067, mae:0.484233558177948
mse:0.4645936191082001, mae:0.48453566431999207
mse:0.471344530582428, mae:0.4888715445995331
mse:0.46640849113464355, mae:0.4846985638141632
```

test_loss(original model):
```
mse:0.582106351852417, mae:0.5587486028671265
mse:0.6263450980186462, mae:0.5919432044029236
mse:0.5486109256744385, mae:0.53843754529953
mse:0.5140747427940369, mae:0.5164958834648132
mse:0.549491822719574, mae:0.5349205732345581
```
We can calculate the average and variance of the 5 iterations:
```
TODO
```

