import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import acf
from scipy import signal
from scipy.signal import savgol_filter
from tqdm import tqdm
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

ban_2nd_step = True
def sliding_window_decomposition(df, target_columns, window_size=24*14, step_size=24*4, period=24,
                                smoothing_method='savgol', trend_window=73, 
                                stl_seasonal=7, stl_robust=False, visualization_path='./visualization/'):
    """
    对时间序列数据使用滑动窗口进行季节性分解，并额外处理趋势使其更平滑
    
    参数:
    df: 输入数据框
    target_columns: 要分解的列
    window_size: 滑动窗口大小，默认14天 (24*14)
    step_size: 窗口滑动步长，默认4天 (24*4)
    period: 季节性周期，默认1天 (24)
    smoothing_method: 平滑方法，推荐'savgol'
    trend_window: 趋势平滑窗口大小
    stl_seasonal: STL季节性滤波器窗口长度
    stl_robust: 是否使用稳健估计
    visualization_path: 可视化结果保存路径
    
    返回:
    包含原始数据和分解组件的增强数据框
    """
    # 创建可视化目录
    os.makedirs(visualization_path, exist_ok=True)
    
    # 检查输入数据
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入数据必须是pandas DataFrame")
    
    if 'date' not in df.columns:
        raise ValueError("输入数据必须包含'date'列")
    
    # 确保日期列是datetime类型
    df['date'] = pd.to_datetime(df['date'])
    
    # 创建一个结果数据框，初始包含原始数据
    result_df = df.copy()
    
    # 为要分解的每一列创建新的特征列
    for col in target_columns:
        result_df[f'{col}_trend'] = np.nan
        result_df[f'{col}_seasonal'] = np.nan
    
    # 计算窗口数和起始位置
    total_rows = len(df)
    
    # 计算窗口起始位置，确保最后一个窗口也是完整的
    window_starts = list(range(0, total_rows - window_size + 1, step_size))
    
    # 如果还有剩余数据，添加最后一个窗口（向前延伸至完整窗口大小）
    if total_rows - (window_starts[-1] + window_size) > 0:
        last_start = total_rows - window_size
        window_starts.append(last_start)
    
    print(f"总数据点: {total_rows}, 窗口大小: {window_size} ({window_size//24}天), 滑动步长: {step_size} ({step_size//24}天)")
    print(f"季节周期: {period} ({period//24}天 + {period%24}小时)")
    print(f"趋势平滑方法: {smoothing_method}, 窗口/参数: {trend_window}")
    print(f"STL参数: seasonal={stl_seasonal}, robust={stl_robust}")
    print(f"将创建 {len(window_starts)} 个窗口进行分解")
    
    # 存储每个点被多少个窗口覆盖
    coverage_count = pd.DataFrame(0, index=df.index, columns=target_columns)
    
    # 记录所有窗口的分解结果
    all_decompositions = {col: [] for col in target_columns}
    
    # 对每个滑动窗口进行分解
    for start_idx in tqdm(window_starts, desc="处理滑动窗口"):
        # 确定窗口结束位置
        end_idx = min(start_idx + window_size, total_rows)
        
        # 如果窗口太小，跳过
        if end_idx - start_idx < period * 2:
            print(f"窗口 {start_idx}-{end_idx} 太小(需要至少 {period * 2} 个点)，跳过")
            continue
        
        # 获取窗口数据
        window_data = df.iloc[start_idx:end_idx].copy()
        window_data.set_index('date', inplace=True)
        
        # 对每个目标列进行分解
        for col in target_columns:
            try:
                # 调整STL参数，确保trend > period且为奇数
                trend_param = max(period + 2, trend_window)
                # 确保是奇数
                if trend_param % 2 == 0:
                    trend_param += 1
                    
                # 确保seasonal参数是正整数且小于数据长度的一半
                seasonal_param = min(stl_seasonal, len(window_data[col]) // 2 - 1)
                seasonal_param = max(3, seasonal_param)  # 至少为3
                
                print(f"调整后的STL参数 - trend: {trend_param}, seasonal: {seasonal_param}, period: {period}")
                
                # 使用STL分解，使用调整后的参数
                stl = STL(window_data[col], period=period, seasonal=seasonal_param, 
                        trend=trend_param, robust=stl_robust)
                decomposition = stl.fit()
                
                # 获取STL分解结果
                trend = pd.Series(decomposition.trend, index=window_data.index)
                seasonal = pd.Series(decomposition.seasonal, index=window_data.index)
                residual = pd.Series(decomposition.resid, index=window_data.index)
                # 检查分解结果是否有效
                if trend.isna().all() or seasonal.isna().all():
                    print(f"警告: 列 {col} 窗口 {start_idx}-{end_idx} 分解结果全为NaN，跳过")
                    continue
                
                # 记录每个窗口的分解结果
                decomp_result = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'trend': trend,
                    'seasonal': seasonal,
                    'residual': residual
                }
                all_decompositions[col].append(decomp_result)
                
                # 更新覆盖计数
                coverage_count.iloc[start_idx:end_idx, coverage_count.columns.get_loc(col)] += 1
                
            except Exception as e:
                print(f"处理列 {col} 的窗口 {start_idx}-{end_idx} 时出错: {e}")
    
    # 可视化覆盖计数
    plt.figure(figsize=(15, 5))
    for col in target_columns:
        plt.plot(range(len(coverage_count)), coverage_count[col].values, label=f'{col}')
    plt.title(f'数据点窗口覆盖次数 (窗口大小:{window_size//24}天, 步长:{step_size//24}天)')
    plt.xlabel('数据点索引')
    plt.ylabel('覆盖窗口数')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(visualization_path, f'coverage_count_p{period}.png'))
    plt.close()
    
    # 合并所有窗口的分解结果
    r_squared_values = {}
    error_stats = {}
    
    for col in target_columns:
        # 检查是否有分解结果
        if len(all_decompositions[col]) == 0:
            print(f"警告: 列 {col} 没有有效的分解结果，跳过")
            continue
            
        # 初始化结果列
        trend_sum = pd.Series(0.0, index=df.index)
        seasonal_sum = pd.Series(0.0, index=df.index)
        residual_sum = pd.Series(0.0, index=df.index)
        
        # 汇总所有窗口的分解结果
        for decomp in all_decompositions[col]:
            start_idx = decomp['start_idx']
            end_idx = decomp['end_idx']
            
            # 获取原始索引
            window_indices = df.index[start_idx:end_idx]
            
            # 将分解结果加到总和中
            trend_values = decomp['trend'].values
            seasonal_values = decomp['seasonal'].values
            residual_values = decomp['residual'].values
            
            # 确保长度匹配
            min_len = min(len(window_indices), len(trend_values))
            
            try:
                trend_sum.iloc[start_idx:start_idx+min_len] += trend_values[:min_len]
                seasonal_sum.iloc[start_idx:start_idx+min_len] += seasonal_values[:min_len]
                residual_sum.iloc[start_idx:start_idx+min_len] += residual_values[:min_len]
            except Exception as e:
                print(f"合并窗口结果时出错 ({col}, {start_idx}-{end_idx}): {e}")
        
        # 计算平均值 (考虑到每个点被多少个窗口覆盖)
        coverage = coverage_count[col].values
        
        # 避免除以零
        coverage_nomask = np.where(coverage > 0, coverage, 1)
        
        # 计算平均值
        trend_avg = trend_sum / coverage_nomask
        seasonal_avg = seasonal_sum / coverage_nomask
        residual_avg = residual_sum / coverage_nomask
        
        # 对未被覆盖的点，使用NaN
        trend_avg = np.where(coverage > 0, trend_avg, np.nan)
        seasonal_avg = np.where(coverage > 0, seasonal_avg, np.nan)
        residual_avg = np.where(coverage > 0, residual_avg, np.nan)
        
        # 填补NaN
        trend_avg = pd.Series(trend_avg).fillna(method='ffill').fillna(method='bfill').values
        seasonal_avg = pd.Series(seasonal_avg).fillna(method='ffill').fillna(method='bfill').values
        residual_avg = pd.Series(residual_avg).fillna(method='ffill').fillna(method='bfill').values
        
        # 对趋势进行额外平滑处理 (仅使用savgol方法)
        if smoothing_method == 'savgol':
            # 使用Savitzky-Golay滤波器对趋势再次平滑
            # 窗口必须是奇数并且小于数据长度
            sg_window = min(trend_window, len(trend_avg) - 1)
            sg_window = sg_window if sg_window % 2 == 1 else sg_window - 1
            
            if sg_window >= 3:  # 确保窗口足够大
                try:
                    smoothed_trend = savgol_filter(trend_avg, sg_window, 3)
                    
                    # 由于平滑改变了趋势，调整残差以保持重构一致性
                    residual_adjusted = residual_avg + (trend_avg - smoothed_trend)
                    
                    # 使用平滑后的趋势
                    trend_avg = smoothed_trend
                    residual_avg = residual_adjusted
                except Exception as e:
                    print(f"Savitzky-Golay平滑失败 ({col}): {e}")
        
        # 存入结果数据框
        result_df[f'{col}_trend'] = trend_avg
        result_df[f'{col}_seasonal'] = seasonal_avg
        
        # 计算重构值 (趋势+季节性)
        reconstructed = trend_avg + seasonal_avg
        
        # 计算重构误差 (仅用于评估与可视化)
        reconstruct_error = result_df[col].values - reconstructed
        
        # 计算拟合优度 (R^2)
        # 创建临时不含NaN的数据表
        mask = ~np.isnan(result_df[col].values) & ~np.isnan(reconstructed)
        if np.sum(mask) > 0:  # 确保有足够的有效数据点
            original = result_df[col].values[mask]
            recon = reconstructed[mask]
            errors = original - recon
            
            ss_total = np.sum((original - np.mean(original))**2)
            ss_error = np.sum(errors**2)
            
            if ss_total > 0:  # 避免除以零
                r_squared = 1 - (ss_error / ss_total)
                r_squared_values[col] = r_squared
                
                # 计算标准化误差统计量
                error_stats[col] = {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'max': np.max(np.abs(errors)),
                    'rmse': np.sqrt(np.mean(errors**2)),
                    'mae': np.mean(np.abs(errors))
                }
                
                print(f"{col} 的拟合优度 (R^2): {r_squared:.4f}")
                print(f"{col} 重构误差 - 平均: {error_stats[col]['mean']:.4f}, 标准差: {error_stats[col]['std']:.4f}, RMSE: {error_stats[col]['rmse']:.4f}")
            else:
                print(f"警告: {col} 原始数据方差为零，无法计算拟合优度")
                r_squared_values[col] = np.nan
                error_stats[col] = {'mean': np.nan, 'std': np.nan, 'max': np.nan, 'rmse': np.nan, 'mae': np.nan}
        else:
            print(f"警告: {col} 没有足够的有效数据来计算拟合优度")
            r_squared_values[col] = np.nan
            error_stats[col] = {'mean': np.nan, 'std': np.nan, 'max': np.nan, 'rmse': np.nan, 'mae': np.nan}
    
        # 可视化分解结果和重构
        r_squared = r_squared_values.get(col, 0)    
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'{col} - 季节性分解结果与重构 (周期:{period}小时, R² = {r_squared:.4f}, 平滑:{smoothing_method})', fontsize=16)
        
        # 原始数据
        plt.subplot(5, 1, 1)
        plt.plot(result_df['date'], result_df[col])
        plt.title('原始数据')
        plt.grid(True)
        
        # 趋势组件
        plt.subplot(5, 1, 2)
        plt.plot(result_df['date'], result_df[f'{col}_trend'])
        plt.title(f'趋势组件 (使用{smoothing_method}平滑)')
        plt.grid(True)
        
        # 季节性组件
        plt.subplot(5, 1, 3)
        plt.plot(result_df['date'], result_df[f'{col}_seasonal'])
        plt.title('季节性组件')
        plt.grid(True)
        
        # 重构数据与原始数据比较
        plt.subplot(5, 1, 4)
        plt.plot(result_df['date'], result_df[col], label='原始数据', alpha=0.7)
        plt.plot(result_df['date'], reconstructed, label='重构数据 (趋势+季节)', linestyle='--')
        plt.title('原始数据 vs 重构数据')
        plt.legend()
        plt.grid(True)
        
        # 重构误差
        plt.subplot(5, 1, 5)
        plt.plot(result_df['date'], reconstruct_error)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'重构误差 (RMSE: {error_stats[col].get("rmse", "N/A"):.4f}, MAE: {error_stats[col].get("mae", "N/A"):.4f})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_path, f'{col}_decomp_recon_p{period}.png'))
        plt.close()
        
        # 放大视图 - 一周的数据
        samples_per_day = 24
        days_to_show = 7
        samples_to_show = samples_per_day * days_to_show
        
        # 找一个中间位置开始，避免边缘
        middle_idx = len(result_df) // 2
        start_idx = middle_idx - samples_to_show // 2
        start_idx = max(0, start_idx)
        end_idx = min(len(result_df), start_idx + samples_to_show)
        
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'{col} - 季节性分解与重构 (周期:{period}小时, 局部一周)', fontsize=16)
        
        # 局部数据
        plt.subplot(5, 1, 1)
        plt.plot(result_df['date'].iloc[start_idx:end_idx], 
                 result_df[col].iloc[start_idx:end_idx])
        plt.title('原始数据 (局部)')
        plt.grid(True)
        
        plt.subplot(5, 1, 2)
        plt.plot(result_df['date'].iloc[start_idx:end_idx], 
                 result_df[f'{col}_trend'].iloc[start_idx:end_idx])
        plt.title('趋势组件 (局部)')
        plt.grid(True)
        
        plt.subplot(5, 1, 3)
        plt.plot(result_df['date'].iloc[start_idx:end_idx], 
                 result_df[f'{col}_seasonal'].iloc[start_idx:end_idx])
        plt.title('季节性组件 (局部)')
        plt.grid(True)
        
        plt.subplot(5, 1, 4)
        plt.plot(result_df['date'].iloc[start_idx:end_idx], 
                 result_df[col].iloc[start_idx:end_idx], 
                 label='原始数据', alpha=0.7)
        plt.plot(result_df['date'].iloc[start_idx:end_idx], 
                 reconstructed[start_idx:end_idx], 
                 label='重构数据', linestyle='--')
        plt.title('原始数据 vs 重构数据 (局部)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(5, 1, 5)
        plt.plot(result_df['date'].iloc[start_idx:end_idx], 
                 reconstruct_error[start_idx:end_idx])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('重构误差 (局部)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_path, f'{col}_zoom_decom_p{period}.png'))
        plt.close()
        
        # 日内季节性模式分析
        if period == 24:  # 如果使用1天周期
            # 创建一天内小时的季节性模式图
            plt.figure(figsize=(15, 8))
            plt.suptitle(f'{col} - 日内季节性模式 (周期:{period}小时)', fontsize=16)
            
            # 准备数据 - 从稳定区域选择
            stable_start = len(result_df) // 3  # 跳过前1/3的数据
            stable_end = stable_start + min(len(result_df) // 3, 24 * 28)  # 取不超过28天的稳定区域
            
            stable_data = result_df.iloc[stable_start:stable_end].copy()
            
            # 按小时聚合
            by_hour = stable_data.groupby(stable_data['date'].dt.hour)
            
            # 计算每小时的平均季节性
            seasonal_by_hour = by_hour[f'{col}_seasonal'].mean()
            seasonal_std = by_hour[f'{col}_seasonal'].std()
            
            # 绘制平均日内季节性模式
            plt.subplot(1, 2, 1)
            plt.plot(range(24), seasonal_by_hour, 'o-', label='平均季节性')
            plt.fill_between(range(24), 
                            seasonal_by_hour - seasonal_std, 
                            seasonal_by_hour + seasonal_std, 
                            alpha=0.2, label='±1标准差')
            plt.title('平均日内季节性模式')
            plt.xlabel('小时')
            plt.ylabel('季节性成分')
            plt.xticks(range(0, 24, 3))
            plt.grid(True)
            plt.legend()
            
            # 热力图 - 按天和小时的季节性
            plt.subplot(1, 2, 2)
            
            # 准备数据
            # 创建星期几和小时的组合索引
            stable_data['dayofweek'] = stable_data['date'].dt.dayofweek
            stable_data['hour'] = stable_data['date'].dt.hour
            
            try:
                pivot_data = stable_data.pivot_table(
                    values=f'{col}_seasonal', 
                    index='dayofweek',
                    columns='hour',
                    aggfunc='mean'
                )
                
                plt.imshow(pivot_data, cmap='RdBu_r', aspect='auto')
                plt.colorbar(label='季节性成分')
                plt.title('按星期几和小时的季节性热力图')
                plt.xlabel('小时')
                plt.ylabel('星期几 (0=周一)')
                plt.xticks(range(0, 24, 3))
                plt.yticks(range(7), ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
            except Exception as e:
                print(f"创建热力图失败: {e}")
                plt.text(0.5, 0.5, '数据不足,无法创建热力图', 
                        horizontalalignment='center', verticalalignment='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_path, f'{col}_daily_pattern_p{period}.png'))
            plt.close()
    
    # 只保留需要的特征
    final_columns = ['date'] + list(df.columns[1:])
    for col in target_columns:
        if f'{col}_trend' in result_df.columns and f'{col}_seasonal' in result_df.columns:
            final_columns.extend([f'{col}_trend', f'{col}_seasonal'])
    
    result_df = result_df[final_columns]
    
    # 可视化重构质量比较
    if len(r_squared_values) > 0:
        metrics = ['rmse', 'mae', 'std']
        width = 0.2
        x = np.arange(len([col for col in target_columns if col in r_squared_values]))
        valid_cols = [col for col in target_columns if col in r_squared_values]
        
        if len(valid_cols) > 0:
            plt.figure(figsize=(12, 8))
            plt.suptitle(f'各特征重构误差对比 (周期:{period}小时)', fontsize=16)
            
            for i, metric in enumerate(metrics):
                values = [error_stats[col][metric] if col in error_stats and metric in error_stats[col] else np.nan for col in valid_cols]
                plt.bar(x + i*width, values, width, label=metric.upper())
            
            plt.title('重构误差统计量比较')
            plt.xlabel('特征')
            plt.ylabel('误差值')
            plt.xticks(x + width, valid_cols)
            plt.legend()
            plt.grid(True, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_path, f'reconstruction_error_p{period}.png'))
            plt.close()
            
            # R^2对比图
            plt.figure(figsize=(12, 6))
            plt.bar(valid_cols, [r_squared_values[col] for col in valid_cols], color='skyblue')
            plt.title(f'各特征的拟合优度 (R²) 对比 (周期:{period}小时)')
            plt.ylabel('R² 值')
            plt.grid(axis='y')
            
            # 为每个条形添加数值标签
            for i, col in enumerate(valid_cols):
                if col in r_squared_values:
                    plt.text(i, r_squared_values[col] + 0.01, f'{r_squared_values[col]:.4f}', 
                            ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_path, f'r_squared_p{period}.png'))
            plt.close()
    
    return result_df, error_stats, r_squared_values

def check_seasonality_in_residual(residual, period=24, threshold=0.3):
    """
    检查残差中是否仍含有季节性
    
    参数:
    residual: 残差序列
    period: 检查的周期
    threshold: 季节性检测阈值
    
    返回:
    boolean: 是否含有季节性
    """
    try:
        # 清理残差中的NaN值
        residual_clean = pd.Series(residual).dropna().values
        
        if len(residual_clean) < period * 2:
            print(f"残差序列长度不足 ({len(residual_clean)} < {period*2})，无法检测季节性")
            return False
            
        # 计算自相关函数
        acf_vals = acf(residual_clean, nlags=period*2, fft=True)
        
        # 查看是否在期望周期处有显著相关
        if np.abs(acf_vals[period]) > threshold:
            return True
            
        # 检查周期图中是否有明显峰值
        freqs, spectrum = signal.periodogram(residual_clean)
        # 转换为周期
        periods = 1 / freqs[1:]  # 跳过0频率
        spectrum = spectrum[1:]
        
        # 查找接近目标周期的峰值
        peak_indices = signal.find_peaks(spectrum)[0]
        for peak_idx in peak_indices:
            if abs(periods[peak_idx] - period) < period * 0.1:  # 允许10%的偏差
                if spectrum[peak_idx] > 0.5 * np.max(spectrum):  # 峰值需足够高
                    return True
        
        return False
    
    except Exception as e:
        print(f"季节性检查失败: {e}")
        return False

def process_with_feature_specific_settings(df, target_columns, visualization_path='./visualization/'):
    """使用特征特定的参数进行处理"""
    
    # 创建输出目录
    os.makedirs(visualization_path, exist_ok=True)
    
    # 复制数据
    result_df = df.copy()
    
    # 处理常规特征
    regular_features = [col for col in target_columns if col != 'LUFL']
    if regular_features:
        print("\n处理常规特征...")
        regular_dir = os.path.join(visualization_path, 'regular_features')
        os.makedirs(regular_dir, exist_ok=True)
        
        regular_data, _, _ = sliding_window_decomposition(
            df, regular_features, 
            window_size=24*14, 
            step_size=24*4, 
            period=24, 
            smoothing_method='savgol', 
            trend_window=73,
            visualization_path=regular_dir
        )
        
        # 将处理结果合并到结果数据框
        for col in regular_features:
            for suffix in ['_trend', '_seasonal']:
                if f'{col}{suffix}' in regular_data.columns:
                    result_df[f'{col}{suffix}'] = regular_data[f'{col}{suffix}']
    
    # 单独处理 LUFL 特征，采用两步分解
    if 'LUFL' in target_columns:
        print("\n特殊处理 LUFL 特征 (两步分解)...")
        
        # 为LUFL创建单独的目录
        lufl_dir = os.path.join(visualization_path, 'LUFL_special')
        os.makedirs(lufl_dir, exist_ok=True)
        
        # 创建只包含LUFL的数据框
        lufl_df = df[['date', 'LUFL']].copy()
        
        try:
            # 第一步：标准STL分解 (日周期)
            print("LUFL 第一步分解: 日周期(24小时)...")
            
            # 检查LUFL是否有nan值，如果有则替换
            if lufl_df['LUFL'].isna().any():
                print(f"警告: LUFL包含 {lufl_df['LUFL'].isna().sum()} 个NaN值，进行填充")
                lufl_df['LUFL'] = lufl_df['LUFL'].fillna(method='ffill').fillna(method='bfill')
            
            lufl_data1, _, _ = sliding_window_decomposition(
                lufl_df, ['LUFL'], 
                window_size=24*14, 
                step_size=24*4, 
                period=24,
                smoothing_method='savgol',  
                trend_window=73,
                visualization_path=os.path.join(lufl_dir, 'step1_day')
            )
            
            # 检查第一步分解是否成功
            if 'LUFL_trend' not in lufl_data1.columns or 'LUFL_seasonal' not in lufl_data1.columns:
                raise ValueError("第一步分解没有生成有效的趋势和季节性组件")
                
            # 计算残差
            residual = lufl_data1['LUFL'] - (lufl_data1['LUFL_trend'] + lufl_data1['LUFL_seasonal'])
            
            # 检查残差是否有效
            if residual.isna().all():
                print("警告: 残差全为NaN，跳过周周期检查")
                weekly_seasonality = False
            else:
                # 检查残差是否还有季节性
                weekly_seasonality = check_seasonality_in_residual(residual.values, period=24*7, threshold=0.2)
            
            if weekly_seasonality and not ban_2nd_step :
                print("LUFL 残差中检测到周周期性，进行第二步分解...")
                
                # 第二步：对残差进行周周期分解
                lufl_resid_df = lufl_df.copy()
                lufl_resid_df['LUFL'] = residual.values
                
                # 填充残差中的NaN
                lufl_resid_df['LUFL'] = lufl_resid_df['LUFL'].fillna(method='ffill').fillna(method='bfill')
                
                # 检查填充后是否仍有NaN
                if lufl_resid_df['LUFL'].isna().any():
                    print(f"警告: 填充后仍有 {lufl_resid_df['LUFL'].isna().sum()} 个NaN值")
                    # 用0填充剩余的NaN值
                    lufl_resid_df['LUFL'] = lufl_resid_df['LUFL'].fillna(0)
                
                lufl_resid_decomp, _, _ = sliding_window_decomposition(
                    lufl_resid_df, ['LUFL'], 
                    window_size=24*14, 
                    step_size=24*4, 
                    period=24*7,  # 使用周周期
                    smoothing_method='savgol',
                    trend_window=73,
                    visualization_path=os.path.join(lufl_dir, 'step2_week')
                )
                
                # 检查第二步分解是否成功
                if 'LUFL_trend' in lufl_resid_decomp.columns and 'LUFL_seasonal' in lufl_resid_decomp.columns:
                    # 合并两次分解结果
                    result_df['LUFL_trend'] = lufl_data1['LUFL_trend'] + lufl_resid_decomp['LUFL_trend']
                    result_df['LUFL_seasonal'] = lufl_data1['LUFL_seasonal'] + lufl_resid_decomp['LUFL_seasonal']
                    
                    # 可视化最终结果
                    plt.figure(figsize=(20, 15))
                    plt.suptitle(f'LUFL - 两步分解结果 (日+周周期)', fontsize=16)
                    
                    # 原始数据
                    plt.subplot(5, 1, 1)
                    plt.plot(result_df['date'], result_df['LUFL'])
                    plt.title('原始数据')
                    plt.grid(True)
                    
                    # 第一步趋势和季节
                    plt.subplot(5, 1, 2)
                    plt.plot(lufl_data1['date'], lufl_data1['LUFL_trend'], label='日周期趋势')
                    plt.plot(lufl_resid_decomp['date'], lufl_resid_decomp['LUFL_trend'], label='周周期趋势')
                    plt.title('趋势组件分解')
                    plt.legend()
                    plt.grid(True)
                    
                    # 第一步和第二步季节
                    plt.subplot(5, 1, 3)
                    plt.plot(lufl_data1['date'], lufl_data1['LUFL_seasonal'], label='日周期季节')
                    plt.plot(lufl_resid_decomp['date'], lufl_resid_decomp['LUFL_seasonal'], label='周周期季节')
                    plt.title('季节性组件分解')
                    plt.legend()
                    plt.grid(True)
                    
                    # 最终趋势和季节
                    plt.subplot(5, 1, 4)
                    plt.plot(result_df['date'], result_df['LUFL_trend'], label='合并趋势')
                    plt.plot(result_df['date'], result_df['LUFL_seasonal'], label='合并季节')
                    plt.title('最终分解结果')
                    plt.legend()
                    plt.grid(True)
                    
                    # 重构与原始数据对比
                    plt.subplot(5, 1, 5)
                    plt.plot(result_df['date'], result_df['LUFL'], label='原始数据', alpha=0.7)
                    plt.plot(result_df['date'], result_df['LUFL_trend'] + result_df['LUFL_seasonal'], 
                            label='重构 (两步分解)', linestyle='--')
                    plt.title('原始数据 vs 两步分解重构')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(lufl_dir, 'LUFL_two_step_decomposition.png'))
                    plt.close()
                    
                    # 局部放大视图
                    samples_per_day = 24
                    days_to_show = 14  # 显示两周
                    samples_to_show = samples_per_day * days_to_show
                    
                    middle_idx = len(result_df) // 2
                    start_idx = middle_idx - samples_to_show // 2
                    start_idx = max(0, start_idx)
                    end_idx = min(len(result_df), start_idx + samples_to_show)
                    
                    plt.figure(figsize=(20, 15))
                    plt.suptitle(f'LUFL - 两步分解结果 (局部视图 - 两周)', fontsize=16)
                    
                    # 原始数据 (局部)
                    plt.subplot(5, 1, 1)
                    plt.plot(result_df['date'].iloc[start_idx:end_idx], 
                            result_df['LUFL'].iloc[start_idx:end_idx])
                    plt.title('原始数据 (局部)')
                    plt.grid(True)
                    
                    # 第一步分解 (局部)
                    plt.subplot(5, 1, 2)
                    plt.plot(lufl_data1['date'].iloc[start_idx:end_idx], 
                            lufl_data1['LUFL_seasonal'].iloc[start_idx:end_idx])
                    plt.title('日周期季节性 (局部)')
                    plt.grid(True)
                    
                    # 第二步分解 (局部)
                    plt.subplot(5, 1, 3)
                    plt.plot(lufl_resid_decomp['date'].iloc[start_idx:end_idx], 
                            lufl_resid_decomp['LUFL_seasonal'].iloc[start_idx:end_idx])
                    plt.title('周周期季节性 (局部)')
                    plt.grid(True)
                    
                    # 最终合并季节性 (局部)
                    plt.subplot(5, 1, 4)
                    plt.plot(result_df['date'].iloc[start_idx:end_idx], 
                            result_df['LUFL_seasonal'].iloc[start_idx:end_idx])
                    plt.title('合并季节性 (局部)')
                    plt.grid(True)
                    
                    # 重构与原始对比 (局部)
                    plt.subplot(5, 1, 5)
                    plt.plot(result_df['date'].iloc[start_idx:end_idx], 
                            result_df['LUFL'].iloc[start_idx:end_idx], 
                            label='原始数据', alpha=0.7)
                    plt.plot(result_df['date'].iloc[start_idx:end_idx], 
                            (result_df['LUFL_trend'] + result_df['LUFL_seasonal']).iloc[start_idx:end_idx], 
                            label='重构数据', linestyle='--')
                    plt.title('原始 vs 重构 (局部)')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(lufl_dir, 'LUFL_two_step_zoom.png'))
                    plt.close()
                else:
                    print("第二步分解失败，使用第一步结果")
                    result_df['LUFL_trend'] = lufl_data1['LUFL_trend']
                    result_df['LUFL_seasonal'] = lufl_data1['LUFL_seasonal']
                
            else:
                print("LUFL 残差中未检测到其他周期性，使用单步分解结果")
                result_df['LUFL_trend'] = lufl_data1['LUFL_trend']
                result_df['LUFL_seasonal'] = lufl_data1['LUFL_seasonal']
            
        except Exception as e:
            print(f"LUFL特殊处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果特殊处理失败，回退到标准方法
            print("回退到标准方法...")
            try:
                lufl_fallback, _, _ = sliding_window_decomposition(
                    lufl_df, ['LUFL'], 
                    window_size=24*14, 
                    step_size=24*4, 
                    period=24,
                    smoothing_method='savgol', 
                    trend_window=73,
                    visualization_path=os.path.join(lufl_dir, 'fallback')
                )
                
                if 'LUFL_trend' in lufl_fallback.columns and 'LUFL_seasonal' in lufl_fallback.columns:
                    result_df['LUFL_trend'] = lufl_fallback['LUFL_trend']
                    result_df['LUFL_seasonal'] = lufl_fallback['LUFL_seasonal']
                else:
                    print("警告: 回退方法也失败了，LUFL将没有分解结果")
            except Exception as e2:
                print(f"回退方法也失败了: {e2}")
                # 最后的备选：使用简单的移动平均作为趋势，原始减趋势作为季节性
                print("使用简单移动平均作为最后的备选...")
                
                # 确保没有NaN
                lufl_clean = lufl_df['LUFL'].fillna(method='ffill').fillna(method='bfill')
                
                # 使用移动平均作为趋势
                window_size = 73 if len(lufl_clean) > 73 else len(lufl_clean) // 2
                trend = lufl_clean.rolling(window=window_size, center=True, min_periods=1).mean()
                
                # 季节性 = 原始 - 趋势
                seasonal = lufl_clean - trend
                
                # 保存结果
                result_df['LUFL_trend'] = trend.values
                result_df['LUFL_seasonal'] = seasonal.values
                
                # 简单可视化
                plt.figure(figsize=(15, 10))
                plt.suptitle('LUFL - 简单移动平均分解 (备选方法)', fontsize=16)
                
                plt.subplot(3, 1, 1)
                plt.plot(lufl_df['date'], lufl_df['LUFL'])
                plt.title('原始数据')
                plt.grid(True)
                
                plt.subplot(3, 1, 2)
                plt.plot(lufl_df['date'], trend)
                plt.title(f'趋势组件 (移动平均 窗口={window_size})')
                plt.grid(True)
                
                plt.subplot(3, 1, 3)
                plt.plot(lufl_df['date'], seasonal)
                plt.title('季节性组件 (原始-趋势)')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(lufl_dir, 'LUFL_fallback_ma.png'))
                plt.close()
    
    # 添加时间特征
    result_df = add_time_features(result_df)
    
    return result_df

def add_time_features(df):
    """添加时间相关特征"""
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    # 提取时间特征
    date_series = df['date']
    df['hour'] = date_series.dt.hour
    df['dayofweek'] = date_series.dt.dayofweek
    df['month'] = date_series.dt.month
    df['day'] = date_series.dt.day
    # df['year'] = date_series.dt.year
    
    # 周期性编码
    # 小时周期性 (24小时)
    hour_in_day = date_series.dt.hour + date_series.dt.minute/60
    df['hour_sin'] = np.sin(2 * np.pi * hour_in_day / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour_in_day / 24)
    
    # 一周中的天周期性 (7天)
    df['day_in_week_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_in_week_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 一月中的天周期性
    # days_in_month = date_series.dt.days_in_month
    # day_in_month = date_series.dt.day
    # df['day_in_month_sin'] = np.sin(2 * np.pi * day_in_month / days_in_month)
    # df['day_in_month_cos'] = np.cos(2 * np.pi * day_in_month / days_in_month)
    
    # 季节特征 (一年中的月份)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def process_ETT_dataset(file_path, out_path=None, visualization_path='./visualization/'):
    """
    处理ETT数据集，添加季节性分解特征和时间特征
    
    参数:
    file_path: ETT数据集文件路径
    out_path: 输出文件路径，默认为None (不保存)
    visualization_path: 可视化结果保存路径
    
    返回:
    增强的数据框
    """
    # 创建可视化目录
    os.makedirs(visualization_path, exist_ok=True)
    
    # 当前时间作为运行标识
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"开始处理，运行ID: {run_id}")
    
    print(f"读取数据: {file_path}")
    data = pd.read_csv(file_path)
    
    # 确保日期列格式正确
    data['date'] = pd.to_datetime(data['date'])
    
    # 获取数据概览
    print(f"原始数据集大小: {data.shape}")
    print(f"日期范围: {data['date'].min()} 到 {data['date'].max()}")
    print(f"列: {data.columns.tolist()}")
    
    # 处理一部分数据
    data = data.iloc[:720*2]  # 处理前720*2个数据点
    
    print(f"截取后的数据集大小: {data.shape}")
    print(f"截取后的日期范围: {data['date'].min()} 到 {data['date'].max()}")
    
    # 数值列
    numeric_cols = [col for col in data.columns if col != 'date']
    
    # 检查数据中是否有NaN值
    for col in numeric_cols:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            print(f"警告: 列 {col} 包含 {nan_count} 个NaN值 ({nan_count/len(data)*100:.2f}%)")
    
    # 可视化原始数据
    plt.figure(figsize=(15, 10))
    plt.suptitle('ETT数据集原始特征', fontsize=16)
    
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(len(numeric_cols), 1, i)
        plt.plot(data['date'], data[col])
        plt.title(f'{col} 原始数据')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_path, 'original_features.png'))
    plt.close()
    
    # 使用特征特定设置进行处理
    print("\n使用特征特定的参数进行处理...")
    enhanced_data = process_with_feature_specific_settings(
        data, numeric_cols, visualization_path
    )
    
    # 保存结果
    if out_path:
        print(f"\n保存增强数据集到: {out_path}")
        enhanced_data.to_csv(out_path, index=False)
    
    print("\nETT数据集处理完成!")
    print(f"最终数据集大小: {enhanced_data.shape}")
    
    # 计算新增特征数
    original_cols = set(['date'] + numeric_cols)
    all_cols = set(enhanced_data.columns)
    new_cols = list(all_cols - original_cols)
    new_cols.sort()
    
    print(f"新增特征数: {len(new_cols)}")
    print(f"新增特征: {new_cols}")
    
    # 检查缺失值
    print("\n检查最终数据集缺失值:")
    for col in enhanced_data.columns:
        nan_count = enhanced_data[col].isna().sum()
        if nan_count > 0:
            print(f"警告: 列 {col} 包含 {nan_count} 个NaN值 ({nan_count/len(enhanced_data)*100:.2f}%)")
            # 填充缺失值
            enhanced_data[col] = enhanced_data[col].fillna(method='ffill').fillna(method='bfill')
    
    # 创建汇总报告
    summary = f"""
    # ETT数据集增强版季节性分解报告
    
    ## 运行信息
    - 运行ID: {run_id}
    - 处理时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    ## 数据概览
    - 原始数据集大小: {data.shape}
    - 日期范围: {data['date'].min()} 到 {data['date'].max()}
    
    ## 季节性分解设置
    - 滑动窗口大小: 24*14 (14 天)
    - 滑动步长: 24*4 (4 天)
    - 常规特征季节周期: 24 (1 天)
    - LUFL特征特殊处理: 两步分解，先日周期(24h)再周周期(24*7h)
    - 趋势平滑方法: savgol
    - 趋势窗口大小: 73
    
    ## 特征汇总
    - 原始特征: {numeric_cols}
    - 新增特征数: {len(new_cols)}
    - 新增特征: {new_cols}
    - 最终数据集大小: {enhanced_data.shape}
    
    ## 主要可视化结果
    - 原始特征图: original_features.png
    - 常规特征处理: regular_features/
    - LUFL特征特殊处理: LUFL_special/
    """
    
    with open(os.path.join(visualization_path, 'decomposition_summary.md'), 'w') as f:
        f.write(summary)
    
    return enhanced_data

# 主函数
if __name__ == "__main__":
    # 替换为实际ETT数据集路径
    file_path = '/mnt/d/周宸源/大学/学习/ML/TSF/Model/datasets/ETTh1.csv'  # 修改为您的文件路径
    out_path = 'ETTh1_enhanced_features.csv'
    
    # 创建带时间戳的可视化目录
    vis_dir = f'./visualization_ETT_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    
    # 处理数据集
    enhanced_data = process_ETT_dataset(file_path, out_path, vis_dir)
    
    print(f"程序执行完成! 可视化结果保存在 {vis_dir}")