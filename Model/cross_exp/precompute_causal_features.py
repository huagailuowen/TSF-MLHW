"""
预计算ETT数据集的因果特征 - 采用滑动窗口平均的方法

此脚本为每个时间点计算特征分解，使用与原始方法一致的滑动窗口平均策略，
但严格保证因果关系，每个时间点只使用该点及之前的数据。
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from scipy.signal import savgol_filter
from tqdm import tqdm
import argparse

# 设置常量
ORIGINAL_FEATURES = 7  # ETT数据集原始特征数量

def precompute_endpoint_features(data_path, output_dir, input_len=720, data_split=None):
    """
    为所有可能成为输入窗口结束点的时刻预计算特征分解结果
    
    参数:
    data_path: 原始数据路径
    output_dir: 输出特征保存目录
    input_len: 模型输入序列长度，同时也是最大历史窗口长度
    data_split: 数据分割点
    """
    print(f"正在读取数据: {data_path}")
    data = pd.read_csv(data_path)
    print(f"数据大小: {data.shape}")
    
    # 提取列名
    cols = list(data.columns)
    if 'date' in cols:
        cols.remove('date')
    original_cols = cols[:ORIGINAL_FEATURES]
    
    # 应用数据分割
    if data_split is None:
        # 默认分割
        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        train_end = num_train
        val_start = train_end 
        val_end = val_start + num_vali
        test_start = val_end
        test_end = len(data)
    else:
        # 使用提供的分割
        train_end = data_split[0]
        val_start = data_split[0]
        val_end = data_split[1]
        test_start = data_split[1]
        test_end = data_split[2]
    
    print(f"数据分割: 训练集 [0:{train_end}], 验证集 [{val_start}:{val_end}], 测试集 [{test_start}:{test_end}]")
    
    # 计算需要处理的范围
    # 为验证集和测试集中可能的输入窗口结束点计算特征
    val_range = range(val_start, val_end)
    test_range = range(test_start, test_end)
    
    # 初始化结果字典，键为时间点，值为特征
    endpoint_features = {}
    
    # 处理验证集
    print(f"\n为验证集计算端点特征 ({len(val_range)}个点)...")
    for t in tqdm(val_range):
        # 在t处计算特征，仅使用最多input_len个历史点
        features_t = compute_features_at_endpoint(data, t, original_cols, input_len)
        endpoint_features[t] = features_t
    
    # 处理测试集
    print(f"\n为测试集计算端点特征 ({len(test_range)}个点)...")
    for t in tqdm(test_range):
        if t not in endpoint_features:  # 避免重复计算
            features_t = compute_features_at_endpoint(data, t, original_cols, input_len)
            endpoint_features[t] = features_t
    
    # 转换为DataFrame并保存
    endpoints = sorted(endpoint_features.keys())
    feature_names = []
    for col in original_cols:
        feature_names.extend([f"{col}_trend", f"{col}_seasonal"])
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(
        [endpoint_features[t] for t in endpoints], 
        index=endpoints,
        columns=feature_names
    )
    
    # 保存到CSV文件
    output_path = os.path.join(output_dir, f'endpoint_features_len{input_len}.csv')
    result_df.to_csv(output_path)
    print(f"已保存端点特征到 {output_path}, 形状: {result_df.shape}")
    
    # 也保存为numpy文件以便快速加载
    output_npy = os.path.join(output_dir, f'endpoint_features_len{input_len}.npy')
    np.save(output_npy, {
        'endpoints': np.array(endpoints),
        'features': np.array([endpoint_features[t] for t in endpoints]),
        'feature_names': feature_names
    })
    print(f"同时保存为numpy格式: {output_npy}")

def compute_features_at_endpoint(data, t, cols, input_len=720):
    """
    计算输入窗口结束点t处的特征，使用滑动窗口分解并平均的方法
    
    参数:
    data: 完整数据
    t: 当前时间点
    cols: 需处理的列名
    input_len: 最大历史窗口长度
    
    返回:
    该时间点的趋势和季节性特征
    """
    # 设置滑动窗口参数，与原始方法一致
    window_size = 14*24  # 14天
    step_size = 4*24     # 4天
    trend_window = 73    # 趋势平滑窗口
    
    # 确保只使用历史数据，最多input_len个点
    max_history = min(t+1, input_len)
    history_start = t+1 - max_history
    
    # 如果历史数据不够一个完整窗口，则直接使用简单方法
    if max_history < window_size:
        return compute_simple_features(data.iloc[history_start:t+1], cols)
    
    # 计算滑动窗口的起始位置
    window_starts = []
    current_start = t - window_size + 1
    
    # 向前移动窗口，确保窗口完全在历史数据内
    while current_start >= history_start:
        window_starts.append(current_start)
        current_start -= step_size
    
    if not window_starts:
        # 如果没有完整窗口，使用最大可能的窗口
        window_starts = [history_start]
    
    print(f"时间点 {t}: 使用 {len(window_starts)} 个滑动窗口进行分解")
    
    # 对每个特征进行分解
    result_features = []
    
    for col in cols:
        # 初始化该特征的滑动窗口分解结果累加器
        trend_sum = 0.0
        seasonal_sum = 0.0
        window_count = 0
        
        # 对每个窗口进行分解
        for start_idx in window_starts:
            end_idx = start_idx + window_size
            
            # 确保窗口不超过当前时间点
            end_idx = min(end_idx, t+1)
            
            # 提取窗口数据
            window_data = data.iloc[start_idx:end_idx][col].values
            
            try:
                # 确保窗口足够长
                if len(window_data) >= 3*24:  # 至少3天数据
                    # STL分解
                    stl = STL(window_data, period=24, seasonal=13, trend=73)
                    result = stl.fit()
                    
                    # 获取当前时间点在窗口中的位置
                    rel_pos = t - start_idx
                    if rel_pos < len(result.trend):
                        trend = result.trend[rel_pos]
                        seasonal = result.seasonal[rel_pos]
                        
                        # 累加分解结果
                        trend_sum += trend
                        seasonal_sum += seasonal
                        window_count += 1
            except Exception as e:
                print(f"处理 {col} 时间点 {t} 窗口 {start_idx}-{end_idx} 出错: {str(e)}")
        
        # 计算平均值
        if window_count > 0:
            trend_avg = trend_sum / window_count
            seasonal_avg = seasonal_sum / window_count
            
            # 对趋势进行Savgol平滑
            try:
                # 需要收集足够的趋势点才能应用平滑
                trend_points = []
                for start_idx in window_starts:
                    end_idx = min(start_idx + window_size, t+1)
                    rel_pos = t - start_idx
                    
                    try:
                        window_data = data.iloc[start_idx:end_idx][col].values
                        if len(window_data) >= 3*24:
                            stl = STL(window_data, period=24, seasonal=13, trend=73)
                            result = stl.fit()
                            if rel_pos < len(result.trend):
                                trend_points.append(result.trend[rel_pos])
                    except:
                        pass
                
                if len(trend_points) >= 3:
                    sg_window = min(trend_window, len(trend_points))
                    sg_window = sg_window if sg_window % 2 == 1 else sg_window - 1
                    if sg_window >= 3:
                        smoothed_trend = savgol_filter(trend_points, sg_window, 3)
                        smoothed_trend_avg = smoothed_trend[-1]  # 使用平滑后的最后一个点
                        
                        # 调整季节性，保持一致性
                        seasonal_adjusted = seasonal_avg + (trend_avg - smoothed_trend_avg)
                        
                        trend_avg = smoothed_trend_avg
                        seasonal_avg = seasonal_adjusted
            except Exception as e:
                print(f"Savgol平滑处理失败: {str(e)}")
            
            # 保存结果
            result_features.extend([trend_avg, seasonal_avg])
        else:
            # 如果没有有效窗口，使用简单方法
            simple_features = compute_simple_features(data.iloc[history_start:t+1], [col])
            result_features.extend(simple_features)
    
    return result_features

def compute_simple_features(data_window, cols):
    """
    当历史数据不足以使用滑动窗口时，使用简单方法计算特征
    """
    result_features = []
    
    for col in cols:
        series = data_window[col].values
        
        if len(series) >= 24:  # 至少有一天的数据
            # 使用简单移动平均作为趋势
            window_size = min(73, len(series) - 1)
            window_size = window_size if window_size % 2 == 1 else window_size - 1
            if window_size >= 3:
                trend = pd.Series(series).rolling(window=window_size, center=True, min_periods=1).mean().iloc[-1]
            else:
                trend = np.mean(series)
                
            # 季节性为原始值减去趋势
            seasonal = series[-1] - trend
        else:
            # 数据太少，趋势就是均值
            trend = np.mean(series)
            seasonal = series[-1] - trend
            
        result_features.extend([trend, seasonal])
    
    return result_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预计算ETT数据集的端点特征')
    parser.add_argument('--data_path', type=str, required=True, help='ETT数据集CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='./', help='输出文件保存目录')
    parser.add_argument('--input_len', type=int, default=720, help='模型输入序列长度，同时也是最大历史窗口长度')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ETT数据集分割 [12*30*24, 4*30*24, 4*30*24] train,val,test
    data_split = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 4*30*24 + 4*30*24]
    
    # 执行预计算
    precompute_endpoint_features(args.data_path, args.output_dir, args.input_len, data_split)
    
    print("\n预计算完成！")