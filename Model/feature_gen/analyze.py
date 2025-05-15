import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import acf, pacf, adfuller
from datetime import datetime
from scipy import signal
from scipy.signal import find_peaks
import scipy.stats as stats  # 这是正确的导入
import traceback  # 添加这个导入
import warnings
warnings.filterwarnings('ignore')

# 读取数据
file_path = '/mnt/d/周宸源/大学/学习/ML/TSF/Model/datasets/ETTh1.csv'
data = pd.read_csv(file_path)

# 数据预处理
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 只选择前7天的数据
samples_per_day = 24  # 假设每天96个样本
days = 7
data = data.iloc[:720]

print(f"分析的数据范围: {data.index.min()} 到 {data.index.max()}")
print(f"总样本数: {len(data)}")

# 显示数据基本信息
print("数据基本信息:")
print(data.head())
print("\n数据统计描述:")
print(data.describe())

# 可视化原始数据
plt.figure(figsize=(15, 10))
for i, col in enumerate(data.columns):
    plt.subplot(len(data.columns), 1, i+1)
    plt.plot(data[col])
    plt.title(col)
    plt.tight_layout()
plt.savefig('raw_data_visualization_7days.png')
plt.close()

# 函数：检查平稳性
def check_stationarity(timeseries, column_name):
    print(f"\n平稳性检验 - {column_name}")
    result = adfuller(timeseries.dropna())
    print('ADF统计量: %f' % result[0])
    print('p-value: %f' % result[1])
    print('临界值:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] <= 0.05:
        print("数据是平稳的")
    else:
        print("数据不是平稳的")
    return result[1] <= 0.05

# 函数：标准季节性分解 - 修复残差问题
def perform_standard_decomposition(series, column_name):
    results = {}
    periods_to_try = [24, 24*7]  # 1天, 12小时, 6小时, 半周
    models = ['additive', 'multiplicative']
    
    for period in periods_to_try:
        for model in models:
            try:
                # 跳过不合适的情况
                if model == 'multiplicative' and (np.any(series <= 0) or np.isnan(series).any()):
                    print(f"跳过乘法模型，因为数据包含非正值")
                    continue
                
                if len(series) < period * 2:
                    print(f"数据长度不足以进行周期为{period}的分解")
                    continue
                
                print(f"执行 {model} 分解，周期 = {period}")
                
                # 执行分解
                decomposition = seasonal_decompose(series, model=model, period=period)
                
                # 直接计算残差以确保正确性
                if model == 'additive':
                    # 确保NaN值处理一致
                    trend = decomposition.trend.fillna(method='bfill').fillna(method='ffill')
                    seasonal = decomposition.seasonal.fillna(method='bfill').fillna(method='ffill')
                    reconstructed = trend + seasonal
                    calculated_resid = series - reconstructed
                else:  # multiplicative
                    trend = decomposition.trend.fillna(method='bfill').fillna(method='ffill')
                    seasonal = decomposition.seasonal.fillna(method='bfill').fillna(method='ffill')
                    reconstructed = trend * seasonal
                    calculated_resid = series / reconstructed
                    
                # 检查残差不是全NaN
                if calculated_resid.isna().all():
                    print(f"警告: {model}_{period} 的残差全为NaN，跳过")
                    continue
                
                # 保存结果，使用计算得到的残差
                key = f"{model}_{period}"
                results[key] = {
                    'trend': decomposition.trend,
                    'seasonal': decomposition.seasonal,
                    'residual': calculated_resid,
                    'reconstructed': reconstructed
                }
                
                # 计算残差统计量
                residual_clean = calculated_resid.dropna()
                residual_std = np.std(residual_clean)
                residual_mean = np.mean(residual_clean)
                residual_range = np.max(residual_clean) - np.min(residual_clean)
                
                # 检查残差是否为常数
                if residual_std < 1e-10:
                    print(f"警告: {key}的残差标准差极小 ({residual_std})，可能是常数")
                    residual_acf_max = 0
                else:
                    try:
                        # 计算残差的自相关
                        residual_acf = acf(residual_clean, nlags=min(96, len(residual_clean)//4), fft=True)
                        residual_acf_max = np.max(np.abs(residual_acf[1:]))  # 忽略lag=0
                    except:
                        print(f"计算残差ACF失败，设置为0")
                        residual_acf_max = 0
                
                results[key]['stats'] = {
                    'residual_std': residual_std,
                    'residual_mean': residual_mean,
                    'residual_range': residual_range,
                    'residual_acf_max': residual_acf_max
                }
                
                # 计算拟合优度
                orig_var = np.var(series.dropna())
                resid_var = np.var(residual_clean)
                fit_quality = 1 - (resid_var / orig_var)
                results[key]['stats']['fit_quality'] = fit_quality
                
                print(f"{key} 分解拟合度: {fit_quality:.4f}, 残差标准差: {residual_std:.4f}")
                
                # 可视化分解结果 - 增强版
                plt.figure(figsize=(15, 12))
                plt.suptitle(f'{column_name} - {model.capitalize()} 分解 (周期={period})', fontsize=16)
                
                # 原始数据
                plt.subplot(511)
                plt.plot(series)
                plt.title('原始数据')
                
                # 趋势
                plt.subplot(512)
                plt.plot(decomposition.trend)
                plt.title('趋势')
                
                # 季节性
                plt.subplot(513)
                plt.plot(decomposition.seasonal)
                plt.title('季节性')
                
                # 残差
                plt.subplot(514)
                plt.plot(calculated_resid)
                plt.title(f'残差 (标准差: {residual_std:.4f}, 均值: {residual_mean:.4f})')
                
                # 原始vs重建
                plt.subplot(515)
                plt.plot(series, label='原始', alpha=0.7)
                plt.plot(reconstructed, label='重建', linestyle='--')
                plt.title(f'原始 vs 重建 (拟合度: {fit_quality:.4f})')
                plt.legend()
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.92)
                plt.savefig(f'{column_name}_{model}_decomp_period{period}.png')
                plt.close()
                
                # 额外的残差分析图
                if residual_std > 1e-10:
                    plt.figure(figsize=(15, 12))
                    plt.suptitle(f'{column_name} - {model} 周期={period} 残差分析', fontsize=16)
                    
                    # 残差的时间序列
                    plt.subplot(321)
                    plt.plot(calculated_resid)
                    plt.title('残差时间序列')
                    
                    # 残差的直方图
                    plt.subplot(322)
                    plt.hist(residual_clean, bins=30, density=True)
                    # 添加正态分布曲线进行比较
                    x = np.linspace(min(residual_clean), max(residual_clean), 100)
                    plt.plot(x, stats.norm.pdf(x, residual_mean, residual_std))  # 使用stats而不是scipy.stats
                    plt.title('残差分布直方图')
                    
                    # 残差的ACF
                    plt.subplot(323)
                    lags = min(48, len(residual_clean)//3)
                    acf_vals = acf(residual_clean, nlags=lags, fft=True)
                    plt.stem(range(len(acf_vals)), acf_vals)
                    plt.axhline(y=0, color='r', linestyle='-')
                    plt.axhline(y=1.96/np.sqrt(len(residual_clean)), color='r', linestyle='--')
                    plt.axhline(y=-1.96/np.sqrt(len(residual_clean)), color='r', linestyle='--')
                    plt.title('残差ACF')
                    
                    # 残差的PACF
                    plt.subplot(324)
                    pacf_vals = pacf(residual_clean, nlags=lags)
                    plt.stem(range(len(pacf_vals)), pacf_vals)
                    plt.axhline(y=0, color='r', linestyle='-')
                    plt.axhline(y=1.96/np.sqrt(len(residual_clean)), color='r', linestyle='--')
                    plt.axhline(y=-1.96/np.sqrt(len(residual_clean)), color='r', linestyle='--')
                    plt.title('残差PACF')
                    
                    # 残差的QQ图
                    plt.subplot(325)
                    stats.probplot(residual_clean, dist="norm", plot=plt)  # 使用stats而不是scipy.stats
                    plt.title('残差正态QQ图')
                    
                    # 残差周期图
                    plt.subplot(326)
                    freq, psd = signal.periodogram(residual_clean)
                    periods = 1 / freq[1:]  # 跳过频率0
                    psd = psd[1:]
                    plt.plot(periods, psd)
                    plt.title('残差周期谱')
                    plt.xlabel('周期 (样本数)')
                    
                    # 标记周期谱中的峰值
                    peaks, _ = find_peaks(psd, height=np.max(psd)/10)
                    if len(peaks) > 0:
                        peak_periods = periods[peaks]
                        plt.scatter(peak_periods, psd[peaks], color='red')
                        for i, p in enumerate(peak_periods):
                            if p < len(residual_clean) / 2 and p > 2:  # 只标记合理的周期
                                plt.annotate(f"{p:.1f}", (p, psd[peaks[i]]))
                    
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.92)
                    plt.savefig(f'{column_name}_{model}_p{period}_resid_analysis.png')
                    plt.close()
                
            except Exception as e:
                print(f"分解失败 {model} 周期={period}: {e}")
                traceback.print_exc()
    
    # 找出最佳分解（基于拟合优度）
    best_key = None
    best_fit = -float('inf')
    
    for key, result in results.items():
        if 'stats' in result:
            stats_results = result['stats']
            fit = stats_results.get('fit_quality', -float('inf'))
            if fit > best_fit:
                best_fit = fit
                best_key = key
    
    if best_key:
        print(f"{column_name} 的最佳分解: {best_key}, 拟合度: {best_fit:.4f}")
        return results[best_key], best_key
    else:
        return None, None

# 函数：增强版STL分解
def perform_stl_decomposition(series, column_name):
    try:
        # 尝试不同的季节性周期
        # STL要求seasonal是奇数，调整周期
        periods = [23, 24*7-1]  # 接近1天、12小时、6小时的奇数
        results = {}
        
        for period in periods:
            if len(series) < period * 2:
                print(f"数据长度不足以进行STL周期为{period}的分解")
                continue
                
            print(f"执行STL分解，周期 = {period}")
            
            try:
                # 用STL进行分解
                stl = STL(series, seasonal=period, period=period, robust=True)
                stl_result = stl.fit()
                
                # 计算拟合优度
                orig_var = np.var(series.dropna())
                resid_var = np.var(stl_result.resid.dropna())
                fit_quality = 1 - (resid_var / orig_var)
                
                print(f"STL周期={period} 拟合度: {fit_quality:.4f}")
                
                # 保存分解结果
                key = f"stl_{period}"
                results[key] = {
                    'decomposition': stl_result,
                    'trend': stl_result.trend,
                    'seasonal': stl_result.seasonal,
                    'residual': stl_result.resid,
                    'fit_quality': fit_quality
                }
                
                # 残差统计量
                residual_clean = stl_result.resid.dropna()
                residual_std = np.std(residual_clean)
                residual_acf = acf(residual_clean, nlags=min(96, len(residual_clean)//4), fft=True)
                residual_acf_max = np.max(np.abs(residual_acf[1:]))
                
                results[key]['stats'] = {
                    'residual_std': residual_std,
                    'residual_acf_max': residual_acf_max,
                    'fit_quality': fit_quality
                }
                
                # 可视化STL分解结果
                plt.figure(figsize=(15, 12))
                plt.suptitle(f'{column_name} - STL 分解 (周期={period})', fontsize=16)
                
                plt.subplot(511)
                plt.plot(series)
                plt.title('原始数据')
                
                plt.subplot(512)
                plt.plot(stl_result.trend)
                plt.title('趋势')
                
                plt.subplot(513)
                plt.plot(stl_result.seasonal)
                plt.title('季节性')
                
                plt.subplot(514)
                plt.plot(stl_result.resid)
                plt.title(f'残差 (标准差: {residual_std:.4f}, ACF最大值: {residual_acf_max:.4f})')
                
                plt.subplot(515)
                reconstructed = stl_result.trend + stl_result.seasonal
                plt.plot(series, label='原始', alpha=0.7)
                plt.plot(reconstructed, label='重建', linestyle='--')
                plt.title(f'原始 vs 重建 (拟合度: {fit_quality:.4f})')
                plt.legend()
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.92)
                plt.savefig(f'{column_name}_stl_p{period}.png')
                plt.close()
                
                # 残差分析图
                plt.figure(figsize=(15, 12))
                plt.suptitle(f'{column_name} - STL 周期={period} 残差分析', fontsize=16)
                
                # 残差的时间序列
                plt.subplot(321)
                plt.plot(stl_result.resid)
                plt.title('残差时间序列')
                
                # 残差的直方图
                plt.subplot(322)
                plt.hist(residual_clean, bins=30, density=True)
                x = np.linspace(min(residual_clean), max(residual_clean), 100)
                plt.plot(x, stats.norm.pdf(x, np.mean(residual_clean), residual_std))  # 使用stats而不是scipy.stats
                plt.title('残差分布直方图')
                
                # 残差的ACF
                plt.subplot(323)
                lags = min(48, len(residual_clean)//3)
                acf_vals = acf(residual_clean, nlags=lags, fft=True)
                plt.stem(range(len(acf_vals)), acf_vals)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.axhline(y=1.96/np.sqrt(len(residual_clean)), color='r', linestyle='--')
                plt.axhline(y=-1.96/np.sqrt(len(residual_clean)), color='r', linestyle='--')
                plt.title('残差ACF')
                
                # 残差的PACF
                plt.subplot(324)
                pacf_vals = pacf(residual_clean, nlags=lags)
                plt.stem(range(len(pacf_vals)), pacf_vals)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.axhline(y=1.96/np.sqrt(len(residual_clean)), color='r', linestyle='--')
                plt.axhline(y=-1.96/np.sqrt(len(residual_clean)), color='r', linestyle='--')
                plt.title('残差PACF')
                
                # 残差的QQ图
                plt.subplot(325)
                stats.probplot(residual_clean, dist="norm", plot=plt)  # 使用stats而不是scipy.stats
                plt.title('残差正态QQ图')
                
                # 残差周期图
                plt.subplot(326)
                freq, psd = signal.periodogram(residual_clean)
                periods = 1 / freq[1:]
                psd = psd[1:]
                plt.plot(periods, psd)
                plt.title('残差周期谱')
                plt.xlabel('周期 (样本数)')
                
                # 标记周期谱中的峰值
                peaks, _ = find_peaks(psd, height=np.max(psd)/10)
                if len(peaks) > 0:
                    peak_periods = periods[peaks]
                    plt.scatter(peak_periods, psd[peaks], color='red')
                    for i, p in enumerate(peak_periods):
                        if p < len(residual_clean) / 2 and p > 2:
                            plt.annotate(f"{p:.1f}", (p, psd[peaks[i]]))
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.92)
                plt.savefig(f'{column_name}_stl_p{period}_resid_analysis.png')
                plt.close()
                
            except Exception as e:
                print(f"STL分解失败，周期={period}: {e}")
                traceback.print_exc()
        
        # 找出最佳STL分解
        best_key = None
        best_fit = -float('inf')
        
        for key, result in results.items():
            fit = result.get('fit_quality', -float('inf'))
            if fit > best_fit:
                best_fit = fit
                best_key = key
        
        if best_key:
            print(f"{column_name} 的最佳STL分解: {best_key}, 拟合度: {best_fit:.4f}")
            best_result = results[best_key]
            
            return {
                'trend': best_result['trend'],
                'seasonal': best_result['seasonal'],
                'residual': best_result['residual'],
                'stats': best_result.get('stats', {})
            }, best_key
        else:
            return None, None
    
    except Exception as e:
        print(f"STL总体分解失败: {e}")
        traceback.print_exc()
        return None, None

# 函数：特征提取 - 重点关注季节性分解和时间特征
def extract_features(series, decomp_results, decomp_type, best_decomp_key=None):
    # 创建一个空的DataFrame来存储特征
    features = pd.DataFrame(index=series.index)
    
    # 添加原始值
    features['value'] = series
    
    # 添加季节性分解的组件
    if decomp_results is not None:
        try:
            # 确保所有分解组件的长度一致，并且与原始数据对齐
            trend = decomp_results.get('trend', pd.Series(index=series.index)).fillna(method='bfill').fillna(method='ffill')
            seasonal = decomp_results.get('seasonal', pd.Series(index=series.index)).fillna(method='bfill').fillna(method='ffill')
            residual = decomp_results.get('residual', pd.Series(index=series.index)).fillna(0)
            
            # 添加基本分解组件
            features['trend'] = trend
            features['seasonal'] = seasonal
            features['residual'] = residual
            
            # 添加派生特征
            features['detrended'] = series - trend
            features['deseasonal'] = series - seasonal
            
            # 如果有季节性强度信息，添加它
            if 'seasonal_strength' in decomp_results:
                features['seasonal_strength'] = decomp_results['seasonal_strength']
            else:
                # 计算季节性强度
                variance_detrended = np.var((series - trend).dropna())
                variance_residual = np.var(residual.dropna())
                if variance_detrended > 0:
                    seasonal_strength = max(0, 1 - variance_residual / variance_detrended)
                    features['seasonal_strength'] = seasonal_strength
            
            # 添加分解类型和周期信息
            if best_decomp_key:
                features['decomp_type'] = decomp_type
                if "_" in best_decomp_key:
                    parts = best_decomp_key.split('_')
                    if len(parts) > 1:
                        features['decomp_period'] = int(parts[-1])
        
        except Exception as e:
            print(f"添加分解特征时出错: {e}")
    
    # 添加关键时间特征
    features['hour'] = features.index.hour
    features['dayofweek'] = features.index.dayofweek
    features['month'] = features.index.month
    features['day'] = features.index.day

    # 添加周期性编码 (sin/cos变换)
    # 小时周期性 (24小时)
    hour_in_day = features.index.hour + features.index.minute / 60
    features['hour_sin'] = np.sin(2 * np.pi * hour_in_day / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour_in_day / 24)
    
    # 一周中的天周期性 (7天)
    features['day_in_week_sin'] = np.sin(2 * np.pi * features.index.dayofweek / 7)
    features['day_in_week_cos'] = np.cos(2 * np.pi * features.index.dayofweek / 7)
    

    # 添加最基本的滞后特征 (只保留lag_1作为关键特征)
    features['lag_1'] = series.shift(1)
    
    # 去除NaN值
    features = features.dropna()
    
    return features

# 主要分析流程
results = {}
features_dict = {}

for column in data.columns:
    print(f"\n{'='*50}")
    print(f"分析 {column} 列...")
    print(f"{'='*50}")
    series = data[column]
    
    # 1. 尝试标准季节性分解 (使用不同周期和模型)
    print("\n执行标准季节性分解...")
    std_decomp_results, best_std_key = perform_standard_decomposition(series, column)
    
    # 2. 尝试STL分解 (更灵活的趋势和季节性)
    print("\n执行STL分解...")
    stl_decomp_results, best_stl_key = perform_stl_decomposition(series, column)
    
    # 3. 选择更好的分解方法 (基于拟合优度)
    best_decomp = None
    best_method = None
    best_key = None
    
    if std_decomp_results and stl_decomp_results:
        # 比较两种方法的拟合优度
        std_fit = std_decomp_results.get('stats', {}).get('fit_quality', 0)
        stl_fit = stl_decomp_results.get('stats', {}).get('fit_quality', 0)
        
        print(f"\n方法对比: 标准分解拟合度={std_fit:.4f}, STL拟合度={stl_fit:.4f}")
        
        if stl_fit > std_fit:
            best_decomp = stl_decomp_results
            best_method = "stl"
            best_key = best_stl_key
            print(f"为 {column} 选择STL分解 (拟合度: {stl_fit:.4f} vs {std_fit:.4f})")
        else:
            best_decomp = std_decomp_results
            best_method = "standard"
            best_key = best_std_key
            print(f"为 {column} 选择标准分解 {best_std_key} (拟合度: {std_fit:.4f} vs {stl_fit:.4f})")
    elif stl_decomp_results:
        best_decomp = stl_decomp_results
        best_method = "stl"
        best_key = best_stl_key
        print(f"为 {column} 选择STL分解 (标准分解失败)")
    elif std_decomp_results:
        best_decomp = std_decomp_results
        best_method = "standard"
        best_key = best_std_key
        print(f"为 {column} 选择标准分解 (STL分解失败)")
    
    # 4. 提取特征
    if best_decomp:
        features = extract_features(series, best_decomp, best_method, best_key)
        features_dict[column] = features
        
        # 查看特征列
        print(f"\n为 {column} 提取的特征列:")
        print(features.columns.tolist())
        print(f"特征形状: {features.shape}")
    
    # 存储结果
    results[column] = {
        'standard_decomp': std_decomp_results,
        'stl_decomp': stl_decomp_results,
        'best_method': best_method,
        'best_key': best_key
    }

# 保存提取的特征
for column, features in features_dict.items():
    features.to_csv(f'{column}_features_decomp.csv')
    print(f"保存了 {column} 的特征到 {column}_features_decomp.csv")

# 创建一个总结报告
report = "# ETTh1 季节性分解分析报告\n\n"
report += "## 数据概览\n"
report += f"- 记录数量: {len(data)}\n"
report += f"- 时间范围: {data.index.min()} 至 {data.index.max()}\n"
report += f"- 变量数量: {len(data.columns)}\n\n"

report += "## 季节性分解结果\n"
for column, result in results.items():
    best_method = result['best_method']
    best_key = result['best_key']
    report += f"### {column}\n"
    report += f"- 最佳分解方法: {best_method}\n"
    if best_method == "standard":
        report += f"- 分解类型: {best_key}\n"
    elif best_method == "stl":
        report += f"- STL参数: {best_key}\n"
    report += "\n"

    # 添加标准分解结果的统计信息
    if result['standard_decomp'] and 'stats' in result['standard_decomp']:
        stats = result['standard_decomp']['stats']
        report += "**标准分解统计:**\n"
        for stat_name, stat_value in stats.items():
            report += f"- {stat_name}: {stat_value:.4f}\n"
        report += "\n"
    
    # 添加STL分解结果的统计信息
    if result['stl_decomp'] and 'stats' in result['stl_decomp']:
        stats = result['stl_decomp']['stats']
        report += "**STL分解统计:**\n"
        for stat_name, stat_value in stats.items():
            report += f"- {stat_name}: {stat_value:.4f}\n"
        report += "\n"
    
    # 添加特征数量信息
    if column in features_dict:
        features = features_dict[column]
        report += f"**提取的特征:**\n"
        report += f"- 特征数量: {len(features.columns)}\n"
        report += f"- 特征列表: {', '.join(features.columns.tolist())}\n\n"
    
    report += "---\n\n"
    
# 保存报告
with open("ETTh1_decomposition_report.md", "w") as f:
    f.write(report)

print("\n分析完成! 结果已保存为图片、CSV文件和分析报告。")