import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def sliding_window_decomposition(df, target_columns, window_size=24*21, step_size=24*7, period=24*7, 
                                visualization_path='./visualization/'):
    """
    对时间序列数据使用滑动窗口进行季节性分解
    
    参数:
    df: 输入数据框
    target_columns: 要分解的列
    window_size: 滑动窗口大小，默认21天 (24*21)
    step_size: 窗口滑动步长，默认7天 (24*7)
    period: 季节性周期，默认一周 (24*7)
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
        result_df[f'{col}_residual'] = np.nan
    
    # 计算窗口数
    total_rows = len(df)
    num_windows = max(1, int((total_rows - window_size) / step_size) + 1)
    
    print(f"总数据点: {total_rows}, 窗口大小: {window_size}, 滑动步长: {step_size}")
    print(f"将创建 {num_windows} 个窗口进行分解")
    
    # 存储每个点被多少个窗口覆盖
    coverage_count = pd.DataFrame(0, index=df.index, columns=target_columns)
    
    # 记录所有窗口的分解结果
    all_decompositions = {col: [] for col in target_columns}
    
    # 对每个滑动窗口进行分解
    for start_idx in tqdm(range(0, total_rows - window_size + 1, step_size), desc="处理滑动窗口"):
        # 确定窗口结束位置
        end_idx = start_idx + window_size
        
        # 如果窗口太小，跳过
        if end_idx - start_idx < period * 2:
            print(f"窗口 {start_idx}-{end_idx} 太小，跳过")
            continue
        
        # 获取窗口数据
        window_data = df.iloc[start_idx:end_idx].copy()
        window_data.set_index('date', inplace=True)
        
        # 对每个目标列进行分解
        for col in target_columns:
            try:
                # 季节性分解，使用加法模型
                decomposition = seasonal_decompose(
                    window_data[col], model='additive', period=period, extrapolate_trend='freq'
                )
                
                # 记录每个窗口的分解结果
                decomp_result = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'trend': decomposition.trend,
                    'seasonal': decomposition.seasonal,
                    'residual': decomposition.resid
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
    plt.title('数据点窗口覆盖次数')
    plt.xlabel('数据点索引')
    plt.ylabel('覆盖窗口数')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(visualization_path, 'coverage_count.png'))
    plt.close()
    
    # 合并所有窗口的分解结果
    for col in target_columns:
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
        
        # 存入结果数据框
        result_df[f'{col}_trend'] = trend_avg
        result_df[f'{col}_seasonal'] = seasonal_avg
        result_df[f'{col}_residual'] = residual_avg
        
        # 计算重构值 (趋势+季节性)
        result_df[f'{col}_reconstructed'] = trend_avg + seasonal_avg
        
        # 计算重构误差
        result_df[f'{col}_reconstruct_error'] = result_df[col] - result_df[f'{col}_reconstructed']
        
        # 计算拟合优度 (R^2)
        ss_total = np.sum((result_df[col].dropna() - result_df[col].dropna().mean())**2)
        ss_error = np.sum((result_df[f'{col}_reconstruct_error'].dropna())**2)
        r_squared = 1 - (ss_error / ss_total)
        
        print(f"{col} 的拟合优度 (R^2): {r_squared:.4f}")
        
        # 计算标准化误差统计量
        mean_error = np.mean(result_df[f'{col}_reconstruct_error'].dropna())
        std_error = np.std(result_df[f'{col}_reconstruct_error'].dropna())
        max_error = np.max(np.abs(result_df[f'{col}_reconstruct_error'].dropna()))
        
        print(f"{col} 重构误差 - 平均: {mean_error:.4f}, 标准差: {std_error:.4f}, 最大: {max_error:.4f}")
    
    # 可视化最终分解结果和重构
    for col in target_columns:
        # 全局可视化 - 原始数据、分解组件和重构
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'{col} - 季节性分解结果与重构 (R² = {r_squared:.4f})', fontsize=16)
        
        # 原始数据
        plt.subplot(5, 1, 1)
        plt.plot(result_df['date'], result_df[col])
        plt.title('原始数据')
        plt.grid(True)
        
        # 趋势组件
        plt.subplot(5, 1, 2)
        plt.plot(result_df['date'], result_df[f'{col}_trend'])
        plt.title('趋势组件')
        plt.grid(True)
        
        # 季节性组件
        plt.subplot(5, 1, 3)
        plt.plot(result_df['date'], result_df[f'{col}_seasonal'])
        plt.title('季节性组件')
        plt.grid(True)
        
        # 重构数据与原始数据比较
        plt.subplot(5, 1, 4)
        plt.plot(result_df['date'], result_df[col], label='原始数据', alpha=0.7)
        plt.plot(result_df['date'], result_df[f'{col}_reconstructed'], label='重构数据 (趋势+季节)', linestyle='--')
        plt.title('原始数据 vs 重构数据')
        plt.legend()
        plt.grid(True)
        
        # 重构误差
        plt.subplot(5, 1, 5)
        plt.plot(result_df['date'], result_df[f'{col}_reconstruct_error'])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'重构误差 (平均: {mean_error:.4f}, 标准差: {std_error:.4f})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_path, f'{col}_decomposition_reconstruction.png'))
        plt.close()
        
        # 放大视图 - 两周期时间窗口
        window_size_days = 2 * period // 24  # 两个季节周期
        samples_to_show = min(2 * period, len(result_df))
        start_idx = min(24 * 30, len(result_df) - samples_to_show)  # 从30天后开始，如果可能
        
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'{col} - 季节性分解与重构 (局部窗口 ~ {window_size_days} 天)', fontsize=16)
        
        # 原始数据 (局部)
        plt.subplot(5, 1, 1)
        plt.plot(result_df['date'].iloc[start_idx:start_idx+samples_to_show], 
                 result_df[col].iloc[start_idx:start_idx+samples_to_show])
        plt.title('原始数据 (局部)')
        plt.grid(True)
        
        # 趋势组件 (局部)
        plt.subplot(5, 1, 2)
        plt.plot(result_df['date'].iloc[start_idx:start_idx+samples_to_show], 
                 result_df[f'{col}_trend'].iloc[start_idx:start_idx+samples_to_show])
        plt.title('趋势组件 (局部)')
        plt.grid(True)
        
        # 季节性组件 (局部)
        plt.subplot(5, 1, 3)
        plt.plot(result_df['date'].iloc[start_idx:start_idx+samples_to_show], 
                 result_df[f'{col}_seasonal'].iloc[start_idx:start_idx+samples_to_show])
        plt.title('季节性组件 (局部)')
        plt.grid(True)
        
        # 重构数据与原始数据比较 (局部)
        plt.subplot(5, 1, 4)
        plt.plot(result_df['date'].iloc[start_idx:start_idx+samples_to_show], 
                 result_df[col].iloc[start_idx:start_idx+samples_to_show], 
                 label='原始数据', alpha=0.7)
        plt.plot(result_df['date'].iloc[start_idx:start_idx+samples_to_show], 
                 result_df[f'{col}_reconstructed'].iloc[start_idx:start_idx+samples_to_show], 
                 label='重构数据', linestyle='--')
        plt.title('原始数据 vs 重构数据 (局部)')
        plt.legend()
        plt.grid(True)
        
        # 重构误差 (局部)
        plt.subplot(5, 1, 5)
        plt.plot(result_df['date'].iloc[start_idx:start_idx+samples_to_show], 
                 result_df[f'{col}_reconstruct_error'].iloc[start_idx:start_idx+samples_to_show])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('重构误差 (局部)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_path, f'{col}_zoom_decom_recon.png'))
        plt.close()
        
        # 季节性模式分析 - 叠加一周的数据
        if period == 24*7:  # 如果使用一周周期
            # 创建各天的季节性模式对比图
            plt.figure(figsize=(15, 8))
            
            # 获取几周的数据，避免边缘效应
            weeks_data = result_df.iloc[24*14:24*28].copy()  # 选择第2-4周的数据
            
            # 按星期几分组
            for day in range(7):
                day_data = weeks_data[weeks_data['date'].dt.dayofweek == day]
                if len(day_data) > 0:
                    plt.subplot(7, 1, day+1)
                    plt.plot(range(24), day_data.groupby(day_data['date'].dt.hour)[f'{col}_seasonal'].mean())
                    plt.title(f'星期{day} 平均季节性模式')
                    plt.grid(True)
                    plt.ylim(weeks_data[f'{col}_seasonal'].min(), weeks_data[f'{col}_seasonal'].max())
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_path, f'{col}_seasonal_pattern_by_weekday.png'))
            plt.close()
    
    # 填充NaN值
    for col in target_columns:
        result_df[f'{col}_trend'] = result_df[f'{col}_trend'].fillna(method='ffill').fillna(method='bfill')
        result_df[f'{col}_seasonal'] = result_df[f'{col}_seasonal'].fillna(method='ffill').fillna(method='bfill')
        result_df[f'{col}_residual'] = result_df[f'{col}_residual'].fillna(method='ffill').fillna(method='bfill')
        result_df[f'{col}_reconstructed'] = result_df[f'{col}_reconstructed'].fillna(method='ffill').fillna(method='bfill')
        result_df[f'{col}_reconstruct_error'] = result_df[f'{col}_reconstruct_error'].fillna(method='ffill').fillna(method='bfill')
    
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
    df['year'] = date_series.dt.year
    
    # 周期性编码
    # 小时周期性 (24小时)
    hour_in_day = date_series.dt.hour + date_series.dt.minute/60
    df['hour_sin'] = np.sin(2 * np.pi * hour_in_day / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour_in_day / 24)
    
    # 一周中的天周期性 (7天)
    df['day_in_week_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_in_week_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 一月中的天周期性
    days_in_month = date_series.dt.days_in_month
    day_in_month = date_series.dt.day
    df['day_in_month_sin'] = np.sin(2 * np.pi * day_in_month / days_in_month)
    df['day_in_month_cos'] = np.cos(2 * np.pi * day_in_month / days_in_month)
    
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
    # data = data.iloc[:int(len(data) * 0.1)]  # 处理10%的数据
    data = data.iloc[:720*2]  # 处理前720*2个数据点
    
    print(f"截取后的数据集大小: {data.shape}")
    print(f"截取后的日期范围: {data['date'].min()} 到 {data['date'].max()}")
    
    # 数值列
    numeric_cols = [col for col in data.columns if col != 'date']
    
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
    
    # 使用滑动窗口分解
    window_size = 24*21  # 3周
    step_size = 24*7     # 1周
    period = 24*7        # 1周
    
    print("\n执行滑动窗口季节性分解...")
    enhanced_data = sliding_window_decomposition(
        data, numeric_cols, window_size, step_size, period, visualization_path
    )
    
    # 添加时间相关特征
    print("\n添加时间特征...")
    enhanced_data = add_time_features(enhanced_data)
    
    # 可视化重构质量比较
    plt.figure(figsize=(15, 10))
    plt.suptitle('各特征重构质量比较', fontsize=16)
    
    # 计算所有特征的重构误差
    error_stats = {}
    for col in numeric_cols:
        errors = enhanced_data[f'{col}_reconstruct_error'].dropna().values
        error_stats[col] = {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'max': np.max(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors**2))
        }
    
    # 绘制误差统计对比图
    metrics = ['mean', 'std', 'max', 'rmse']
    width = 0.2
    x = np.arange(len(numeric_cols))
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        values = [error_stats[col][metric] for col in numeric_cols]
        plt.bar(x + i*width, values, width, label=f'{metric}')
    
    plt.title('重构误差统计量比较')
    plt.xlabel('特征')
    plt.ylabel('误差值')
    plt.xticks(x + width*1.5, numeric_cols)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_path, 'reconstruction_error_comparison.png'))
    plt.close()
    
    # 保存结果
    if out_path:
        print(f"\n保存增强数据集到: {out_path}")
        enhanced_data.to_csv(out_path, index=False)
    
    print("\nETT数据集处理完成!")
    print(f"最终数据集大小: {enhanced_data.shape}")
    new_cols = [col for col in enhanced_data.columns if col not in data.columns]
    print(f"新增特征数: {len(new_cols)}")
    print(f"新增特征: {new_cols}")
    
    # 创建汇总报告
    summary = f"""
    # ETT数据集滑动窗口季节性分解报告
    
    ## 运行信息
    - 运行ID: {run_id}
    - 处理时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    ## 数据概览
    - 原始数据集大小: {data.shape}
    - 日期范围: {data['date'].min()} 到 {data['date'].max()}
    
    ## 季节性分解设置
    - 滑动窗口大小: {window_size} (约 {window_size//24} 天)
    - 滑动步长: {step_size} (约 {step_size//24} 天)
    - 季节周期: {period} (约 {period//24} 天)
    
    ## 重构质量评估
    """
    
    for col in numeric_cols:
        summary += f"""
    ### {col} 重构分析
    - 平均误差: {error_stats[col]['mean']:.6f}
    - 误差标准差: {error_stats[col]['std']:.6f}
    - 最大绝对误差: {error_stats[col]['max']:.6f}
    - RMSE: {error_stats[col]['rmse']:.6f}
    """
    
    summary += f"""
    ## 特征汇总
    - 原始特征: {numeric_cols}
    - 新增特征数: {len(new_cols)}
    - 新增特征: {new_cols}
    - 最终数据集大小: {enhanced_data.shape}
    
    ## 主要可视化结果
    - 原始特征图: original_features.png
    - 窗口覆盖图: coverage_count.png
    - 各特征完整分解与重构图: <feature>_decomposition_reconstruction.png
    - 各特征局部分解与重构图: <feature>_zoom_decom_recon.png
    - 季节性模式按星期几分析: <feature>_seasonal_pattern_by_weekday.png (如适用)
    - 重构误差比较: reconstruction_error_comparison.png
    """
    
    with open(os.path.join(visualization_path, 'decomposition_summary.md'), 'w') as f:
        f.write(summary)
    
    return enhanced_data

# 主函数
if __name__ == "__main__":
    # 替换为实际ETT数据集路径
    file_path = '/mnt/d/周宸源/大学/学习/ML/TSF/Model/datasets/ETTh1.csv'  # 修改为您的文件路径
    out_path = 'ETTh1_enhanced.csv'
    
    # 创建带时间戳的可视化目录
    vis_dir = f'./visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    
    # 处理数据集
    enhanced_data = process_ETT_dataset(file_path, out_path, vis_dir)
    
    print(f"程序执行完成! 可视化结果保存在 {vis_dir}")