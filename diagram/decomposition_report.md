分析的数据范围: 2016-07-01 00:00:00 到 2016-07-30 23:00:00
总样本数: 720
数据基本信息:
                      HUFL   HULL   MUFL   MULL   LUFL   LULL         OT
date                                                                    
2016-07-01 00:00:00  5.827  2.009  1.599  0.462  4.203  1.340  30.531000
2016-07-01 01:00:00  5.693  2.076  1.492  0.426  4.142  1.371  27.787001
2016-07-01 02:00:00  5.157  1.741  1.279  0.355  3.777  1.218  27.787001
2016-07-01 03:00:00  5.090  1.942  1.279  0.391  3.807  1.279  25.044001
2016-07-01 04:00:00  5.358  1.942  1.492  0.462  3.868  1.279  21.948000

数据统计描述:
             HUFL        HULL        MUFL        MULL        LUFL        LULL          OT
count  720.000000  720.000000  720.000000  720.000000  720.000000  720.000000  720.000000
mean    11.582397    3.362900    7.893367    1.337039    3.599876    1.397640   33.655460
std      3.300082    1.448346    2.524421    1.186030    1.182476    0.358386    5.971497
min      3.215000   -0.402000    0.107000   -2.168000    1.371000    0.670000   16.882999
25%      9.377000    2.260250    6.396000    0.604000    2.733500    1.127000   29.968000
50%     11.186000    3.215000    7.640000    1.244000    3.290000    1.371000   33.731499
75%     13.597000    4.421000    9.594000    2.239000    4.295000    1.675000   38.339001
max     21.568001    8.841000   16.417000    4.193000    7.889000    3.046000   46.007000

==================================================
分析 HUFL 列...
==================================================

执行标准季节性分解...
执行 additive 分解，周期 = 24
additive_24 分解拟合度: 0.7471, 残差标准差: 1.6583
执行 multiplicative 分解，周期 = 24
multiplicative_24 分解拟合度: 0.9982, 残差标准差: 0.1405
执行 additive 分解，周期 = 168
additive_168 分解拟合度: 0.5778, 残差标准差: 2.1428
执行 multiplicative 分解，周期 = 168
multiplicative_168 分解拟合度: 0.9967, 残差标准差: 0.1906
HUFL 的最佳分解: multiplicative_24, 拟合度: 0.9982

执行STL分解...
执行STL分解，周期 = 23
STL周期=23 拟合度: 0.7047
执行STL分解，周期 = 167
STL周期=167 拟合度: 0.6892
HUFL 的最佳STL分解: stl_23, 拟合度: 0.7047

方法对比: 标准分解拟合度=0.9982, STL拟合度=0.7047
为 HUFL 选择标准分解 multiplicative_24 (拟合度: 0.9982 vs 0.7047)

为 HUFL 提取的特征列:
['value', 'trend', 'seasonal', 'residual', 'detrended', 'deseasonal', 'seasonal_strength', 'decomp_type', 'decomp_period', 'hour', 'dayofweek', 'month', 'day', 'hour_sin', 'hour_cos', 'day_in_week_sin', 'day_in_week_cos', 'lag_1']
特征形状: (719, 18)

==================================================
分析 HULL 列...
==================================================

执行标准季节性分解...
执行 additive 分解，周期 = 24
additive_24 分解拟合度: 0.7702, 残差标准差: 0.6939
跳过乘法模型，因为数据包含非正值
执行 additive 分解，周期 = 168
additive_168 分解拟合度: 0.5660, 残差标准差: 0.9535
跳过乘法模型，因为数据包含非正值
HULL 的最佳分解: additive_24, 拟合度: 0.7702

执行STL分解...
执行STL分解，周期 = 23
STL周期=23 拟合度: 0.6793
执行STL分解，周期 = 167
STL周期=167 拟合度: 0.6207
HULL 的最佳STL分解: stl_23, 拟合度: 0.6793

方法对比: 标准分解拟合度=0.7702, STL拟合度=0.6793
为 HULL 选择标准分解 additive_24 (拟合度: 0.7702 vs 0.6793)

为 HULL 提取的特征列:
['value', 'trend', 'seasonal', 'residual', 'detrended', 'deseasonal', 'seasonal_strength', 'decomp_type', 'decomp_period', 'hour', 'dayofweek', 'month', 'day', 'hour_sin', 'hour_cos', 'day_in_week_sin', 'day_in_week_cos', 'lag_1']
特征形状: (719, 18)

==================================================
分析 MUFL 列...
==================================================

执行标准季节性分解...
执行 additive 分解，周期 = 24
additive_24 分解拟合度: 0.7238, 残差标准差: 1.3259
执行 multiplicative 分解，周期 = 24
multiplicative_24 分解拟合度: 0.9953, 残差标准差: 0.1730
执行 additive 分解，周期 = 168
additive_168 分解拟合度: 0.5275, 残差标准差: 1.7341
执行 multiplicative 分解，周期 = 168
multiplicative_168 分解拟合度: 0.9909, 残差标准差: 0.2409
MUFL 的最佳分解: multiplicative_24, 拟合度: 0.9953

执行STL分解...
执行STL分解，周期 = 23
STL周期=23 拟合度: 0.6594
执行STL分解，周期 = 167
STL周期=167 拟合度: 0.6814
MUFL 的最佳STL分解: stl_167, 拟合度: 0.6814

方法对比: 标准分解拟合度=0.9953, STL拟合度=0.6814
为 MUFL 选择标准分解 multiplicative_24 (拟合度: 0.9953 vs 0.6814)

为 MUFL 提取的特征列:
['value', 'trend', 'seasonal', 'residual', 'detrended', 'deseasonal', 'seasonal_strength', 'decomp_type', 'decomp_period', 'hour', 'dayofweek', 'month', 'day', 'hour_sin', 'hour_cos', 'day_in_week_sin', 'day_in_week_cos', 'lag_1']
特征形状: (719, 18)

==================================================
分析 MULL 列...
==================================================

执行标准季节性分解...
执行 additive 分解，周期 = 24
additive_24 分解拟合度: 0.7831, 残差标准差: 0.5519
跳过乘法模型，因为数据包含非正值
执行 additive 分解，周期 = 168
additive_168 分解拟合度: 0.5907, 残差标准差: 0.7583
跳过乘法模型，因为数据包含非正值
MULL 的最佳分解: additive_24, 拟合度: 0.7831

执行STL分解...
执行STL分解，周期 = 23
STL周期=23 拟合度: 0.6821
执行STL分解，周期 = 167
STL周期=167 拟合度: 0.6171
MULL 的最佳STL分解: stl_23, 拟合度: 0.6821

方法对比: 标准分解拟合度=0.7831, STL拟合度=0.6821
为 MULL 选择标准分解 additive_24 (拟合度: 0.7831 vs 0.6821)

为 MULL 提取的特征列:
['value', 'trend', 'seasonal', 'residual', 'detrended', 'deseasonal', 'seasonal_strength', 'decomp_type', 'decomp_period', 'hour', 'dayofweek', 'month', 'day', 'hour_sin', 'hour_cos', 'day_in_week_sin', 'day_in_week_cos', 'lag_1']
特征形状: (719, 18)

==================================================
分析 LUFL 列...
==================================================

执行标准季节性分解...
执行 additive 分解，周期 = 24
additive_24 分解拟合度: 0.6506, 残差标准差: 0.6985
执行 multiplicative 分解，周期 = 24
multiplicative_24 分解拟合度: 0.9759, 残差标准差: 0.1835
执行 additive 分解，周期 = 168
additive_168 分解拟合度: 0.5172, 残差标准差: 0.8211
执行 multiplicative 分解，周期 = 168
multiplicative_168 分解拟合度: 0.9662, 残差标准差: 0.2172
LUFL 的最佳分解: multiplicative_24, 拟合度: 0.9759

执行STL分解...
执行STL分解，周期 = 23
STL周期=23 拟合度: 0.6269
执行STL分解，周期 = 167
STL周期=167 拟合度: 0.4680
LUFL 的最佳STL分解: stl_23, 拟合度: 0.6269

方法对比: 标准分解拟合度=0.9759, STL拟合度=0.6269
为 LUFL 选择标准分解 multiplicative_24 (拟合度: 0.9759 vs 0.6269)

为 LUFL 提取的特征列:
['value', 'trend', 'seasonal', 'residual', 'detrended', 'deseasonal', 'seasonal_strength', 'decomp_type', 'decomp_period', 'hour', 'dayofweek', 'month', 'day', 'hour_sin', 'hour_cos', 'day_in_week_sin', 'day_in_week_cos', 'lag_1']
特征形状: (719, 18)

==================================================
分析 LULL 列...
==================================================

执行标准季节性分解...
执行 additive 分解，周期 = 24
additive_24 分解拟合度: 0.7634, 残差标准差: 0.1742
执行 multiplicative 分解，周期 = 24
multiplicative_24 分解拟合度: 0.8677, 残差标准差: 0.1303
执行 additive 分解，周期 = 168
additive_168 分解拟合度: 0.6100, 残差标准差: 0.2236
执行 multiplicative 分解，周期 = 168
multiplicative_168 分解拟合度: 0.7872, 残差标准差: 0.1652
LULL 的最佳分解: multiplicative_24, 拟合度: 0.8677

执行STL分解...
执行STL分解，周期 = 23
STL周期=23 拟合度: 0.7512
执行STL分解，周期 = 167
STL周期=167 拟合度: 0.5904
LULL 的最佳STL分解: stl_23, 拟合度: 0.7512

方法对比: 标准分解拟合度=0.8677, STL拟合度=0.7512
为 LULL 选择标准分解 multiplicative_24 (拟合度: 0.8677 vs 0.7512)

为 LULL 提取的特征列:
['value', 'trend', 'seasonal', 'residual', 'detrended', 'deseasonal', 'seasonal_strength', 'decomp_type', 'decomp_period', 'hour', 'dayofweek', 'month', 'day', 'hour_sin', 'hour_cos', 'day_in_week_sin', 'day_in_week_cos', 'lag_1']
特征形状: (719, 18)

==================================================
分析 OT 列...
==================================================

执行标准季节性分解...
执行 additive 分解，周期 = 24
additive_24 分解拟合度: 0.8939, 残差标准差: 1.9435
执行 multiplicative 分解，周期 = 24
multiplicative_24 分解拟合度: 0.9999, 残差标准差: 0.0668
执行 additive 分解，周期 = 168
additive_168 分解拟合度: 0.7811, 残差标准差: 2.7919
执行 multiplicative 分解，周期 = 168
multiplicative_168 分解拟合度: 0.9998, 残差标准差: 0.0928
OT 的最佳分解: multiplicative_24, 拟合度: 0.9999

执行STL分解...
执行STL分解，周期 = 23
STL周期=23 拟合度: 0.8890
执行STL分解，周期 = 167
STL周期=167 拟合度: 0.8474
OT 的最佳STL分解: stl_23, 拟合度: 0.8890

方法对比: 标准分解拟合度=0.9999, STL拟合度=0.8890
为 OT 选择标准分解 multiplicative_24 (拟合度: 0.9999 vs 0.8890)

为 OT 提取的特征列:
['value', 'trend', 'seasonal', 'residual', 'detrended', 'deseasonal', 'seasonal_strength', 'decomp_type', 'decomp_period', 'hour', 'dayofweek', 'month', 'day', 'hour_sin', 'hour_cos', 'day_in_week_sin', 'day_in_week_cos', 'lag_1']
特征形状: (719, 18)
保存了 HUFL 的特征到 HUFL_features_decomp.csv
保存了 HULL 的特征到 HULL_features_decomp.csv
保存了 MUFL 的特征到 MUFL_features_decomp.csv
保存了 MULL 的特征到 MULL_features_decomp.csv
保存了 LUFL 的特征到 LUFL_features_decomp.csv
保存了 LULL 的特征到 LULL_features_decomp.csv
保存了 OT 的特征到 OT_features_decomp.csv

分析完成! 结果已保存为图片、CSV文件和分析报告。