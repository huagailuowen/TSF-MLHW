import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch

# 设置全局变量表示原始特征数量
ORIGINAL_FEATURES = 7  # ETT数据集的原始变量数量

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, 
                 data_split=None, scale=True, scale_statistic=None,
                 use_precomputed_features=False):
        # size [seq_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        self.scale_statistic = scale_statistic
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.use_precomputed_features = use_precomputed_features
        self.__read_data__()
        
        # 如果需要，加载端点特征并替换特征
        if self.use_precomputed_features and flag in ['test', 'val']:
            self._replace_with_endpoint_features()
            
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # 确保数据齐全
        cols = list(df_raw.columns) 
        if 'date' in cols:
            cols.remove('date')
        data = df_raw[cols].values
        
        # 根据设置进行数据分割
        df_stamp = df_raw[['date']] if 'date' in df_raw.columns else None
        
        # 使用预设的train/val/test分割
        if self.data_split is not None:
            border1s = [0, self.data_split[0], self.data_split[1]]
            border2s = [self.data_split[0], self.data_split[1], self.data_split[2]]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        # 默认分割
        else:
            num_train = int(len(data)*0.7)
            num_test = int(len(data)*0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train, num_train+num_vali]
            border2s = [num_train, num_train+num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
        # 处理边界并提取子集
        if self.set_type == 0: # 训练集
            border2 = border2 - self.seq_len - self.pred_len + 1
        elif self.set_type == 1: # 验证集
            border1 = border1 - self.seq_len
            border2 = border2 - self.seq_len - self.pred_len + 1
        else: # 测试集
            border1 = border1 - self.seq_len
            border2 = border2 - self.seq_len - self.pred_len + 1
            
        # 特征标准化
        if self.scale:
            # 如果提供了统计信息，使用它们
            if self.scale_statistic:
                mean = self.scale_statistic['mean']
                std = self.scale_statistic['std']
                train_data = data[border1s[0]:border2s[0]]
                data = (data - mean) / std
                self.scaler.mean_ = mean
                self.scaler.scale_ = std
            # 否则，使用训练集计算缩放参数
            else:
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(data)
            
        # 提取输入输出序列
        data_x = []
        data_y = []
        
        # 记录每个样本对应的输入窗口范围
        self.input_windows = []
        
        for i in range(border1, border2):
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len
            
            data_x.append(data[s_begin:s_end])
            data_y.append(data[r_begin:r_end])
            
            # 记录每个样本的输入窗口索引
            self.input_windows.append((s_begin, s_end-1))  # 记录开始和结束索引
            
        self.data_x = np.array(data_x)
        self.data_y = np.array(data_y)
        
        # 记录原始数据边界
        self.border1s = border1s
        self.border2s = border2s
        self.border1 = border1
        self.border2 = border2
        
        # 记录日期信息 (如果有)
        self.df_stamp = df_stamp
    
    def _replace_with_endpoint_features(self):
        """
        使用预计算的端点特征替换特征
        """
        endpoint_feature_path = os.path.join(self.root_path, f'endpoint_features_len{self.seq_len}.npy')
        
        if not os.path.exists(endpoint_feature_path):
            print(f"警告: 未找到端点特征文件 {endpoint_feature_path}")
            return
            
        try:
            # 加载端点特征
            print(f"加载端点特征: {endpoint_feature_path}")
            endpoint_data = np.load(endpoint_feature_path, allow_pickle=True).item()
            endpoints = endpoint_data['endpoints']
            features = endpoint_data['features']
            
            # 创建端点到特征的映射
            endpoint_to_feature = {endpoints[i]: features[i] for i in range(len(endpoints))}
            
            # 判断原始特征维度
            _, _, feature_dim = self.data_x.shape
            
            # 检查是否有足够的特征列
            has_extra_features = feature_dim > ORIGINAL_FEATURES
            has_time_features = feature_dim > ORIGINAL_FEATURES + ORIGINAL_FEATURES * 2
            
            if has_extra_features:
                # 提取原始特征部分和时间特征部分（如果有）
                orig_features = self.data_x[:, :, :ORIGINAL_FEATURES]
                
                if has_time_features:
                    time_features = self.data_x[:, :, ORIGINAL_FEATURES + ORIGINAL_FEATURES*2:]
                
                # 创建新的特征数组
                batch_size, seq_len = self.data_x.shape[:2]
                trend_seasonal_features = np.zeros((batch_size, seq_len, ORIGINAL_FEATURES * 2))
                
                # 对每个样本的每个时间点，找到对应的端点特征
                for i, (window_start, window_end) in enumerate(self.input_windows):
                    # 获取窗口结束点对应的特征
                    if window_end in endpoint_to_feature:
                        # 对整个窗口应用相同的特征
                        # 这是合理的，因为在实际预测时，整个窗口会使用相同的分解结果
                        endpoint_feature = endpoint_to_feature[window_end]
                        trend_seasonal_features[i, :, :] = endpoint_feature
                    else:
                        print(f"警告: 未找到端点 {window_end} 的特征")
                
                # 重组特征
                if has_time_features:
                    self.data_x = np.concatenate([orig_features, trend_seasonal_features, time_features], axis=2)
                else:
                    self.data_x = np.concatenate([orig_features, trend_seasonal_features], axis=2)
                    
                print(f"已成功替换为端点特征，新数据形状: {self.data_x.shape}")
            else:
                print("原始数据不包含额外特征列，无法替换")
                
        except Exception as e:
            print(f"加载或替换端点特征时出错: {str(e)}")
    
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)