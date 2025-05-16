from data.data_loader import Dataset_MTS
from cross_exp.exp_basic import Exp_Basic
from cross_models.cross_former import Crossformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import time
import json
import pickle
from statsmodels.tsa.seasonal import STL
from scipy.signal import savgol_filter

import warnings
warnings.filterwarnings('ignore')

# 设置全局变量表示原始特征数量
ORIGINAL_FEATURES = 7  # ETT数据集的原始变量数量

class Exp_crossformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_crossformer, self).__init__(args)
        # 添加新参数，决定是否只用原始特征计算训练损失
        self.train_original_only = args.train_original_only if hasattr(args, 'train_original_only') else False
        
        # 特征生成模式: 'none', 'precomputed', 'online'
        self.feature_mode = args.feature_mode if hasattr(args, 'feature_mode') else 'online'
        
        # 是否在线生成特征时保存到文件
        self.save_online_features = args.save_online_features if hasattr(args, 'save_online_features') else False
    
    def _build_model(self):        
        model = Crossformer(
            self.args.data_dim, 
            self.args.in_len, 
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model, 
            self.args.d_ff,
            self.args.n_heads, 
            self.args.e_layers,
            self.args.dropout, 
            self.args.baseline,
            self.device
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
        
        # ETT数据集分割 [12*30*24, 4*30*24, 4*30*24] train,val,test 
        data_split = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 4*30*24 + 4*30*24]
        
        # 创建数据集时增加参数，指示是否使用预计算的特征
        use_precomputed = (self.feature_mode == 'precomputed' and flag in ['test', 'val'])
        
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.in_len, args.out_len],  
            data_split=data_split,
            use_precomputed_features=use_precomputed
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def _get_features_for_training(self, pred, true):
        """根据设置决定训练时用哪些特征计算损失"""
        if self.train_original_only:
            return pred[:, :, :ORIGINAL_FEATURES], true[:, :, :ORIGINAL_FEATURES]
        return pred, true

    def _get_features_for_evaluation(self, pred, true):
        """评估时永远只使用原始特征"""
        return pred[:, :, :ORIGINAL_FEATURES], true[:, :, :ORIGINAL_FEATURES]
    
    def _sliding_window_decomposition_for_single_series(self, ts, period=24):
        """
        对单个时间序列进行因果滑动窗口季节性分解
        正确实现滑动窗口策略，固定步长移动
        
        参数:
        ts: 单个时间序列，长度为seq_len
        period: 季节周期，默认为24小时
        
        返回:
        trend: 趋势组件
        seasonal: 季节性组件
        """
        seq_len = len(ts)
        
        # 设定参数与原始脚本一致
        window_size = 14*24  # 14天
        step_size = 4*24     # 4天
        trend_window = 73    # 趋势平滑窗口
        
        # 初始化结果数组
        trend_sum = np.zeros(seq_len)
        seasonal_sum = np.zeros(seq_len)
        coverage_count = np.zeros(seq_len)
        
        # 计算最小有效窗口
        min_window = period * 5  # 至少5个周期
        
        # 创建窗口列表
        window_starts = []
        
        # 从末尾向前创建窗口
        current_end = seq_len
        while current_end > min_window:
            # 获取窗口范围
            window_start = max(0, current_end - window_size)
            window_length = current_end - window_start
            
            # 如果窗口足够大，添加到列表
            if window_length >= min_window:
                window_starts.append(window_start)
            
            # 向前移动步长
            current_end -= step_size
        
        window_starts.append(0)  # 确保包含第一个窗口
        # 如果没有窗口，但序列足够长，至少使用一个窗口
        if not window_starts and seq_len >= min_window:
            window_starts = [max(0, seq_len - window_size)]
        
        # 处理每个窗口
        for start_idx in window_starts:
            end_idx = min(start_idx + window_size, seq_len)
            
            # 提取窗口数据
            window_data = ts[start_idx:end_idx]
            
            try:
                # STL分解
                stl = STL(window_data, period=period, seasonal=13, trend=trend_window)
                result = stl.fit()
                
                # 将结果应用于窗口覆盖的每个点
                for i in range(len(result.trend)):
                    if start_idx + i < seq_len:  # 确保不超出原序列
                        trend_sum[start_idx + i] += result.trend[i]
                        seasonal_sum[start_idx + i] += result.seasonal[i]
                        coverage_count[start_idx + i] += 1
            except Exception as e:
                # 如果STL失败，尝试简单方法
                if len(window_data) > period:
                    try:
                        # 使用移动平均作为趋势
                        ma_window = min(trend_window, len(window_data))
                        ma_window = ma_window if ma_window % 2 == 1 else ma_window - 1
                        
                        if ma_window >= 3:
                            trend_vals = pd.Series(window_data).rolling(window=ma_window, center=True, min_periods=1).mean().values
                            # 季节性为原始值减去趋势
                            seasonal_vals = window_data - trend_vals
                            
                            # 将结果应用于窗口覆盖的每个点
                            for i in range(len(trend_vals)):
                                if start_idx + i < seq_len:
                                    trend_sum[start_idx + i] += trend_vals[i]
                                    seasonal_sum[start_idx + i] += seasonal_vals[i]
                                    coverage_count[start_idx + i] += 1
                    except:
                        pass
        
        # 计算平均值，处理没有覆盖的点
        mask = coverage_count > 0
        trend = np.zeros(seq_len)
        seasonal = np.zeros(seq_len)
        
        trend[mask] = trend_sum[mask] / coverage_count[mask]
        seasonal[mask] = seasonal_sum[mask] / coverage_count[mask]
        
        # 对未被覆盖的点进行前向填充
        last_valid = 0
        for i in range(seq_len):
            if coverage_count[i] > 0:
                last_valid = i
            elif i > 0 and last_valid > 0:
                trend[i] = trend[last_valid]
                seasonal[i] = seasonal[last_valid]
        
        # 使用Savitzky-Golay滤波器进行额外平滑 - 与原始脚本一致
        try:
            # 确保窗口大小有效
            sg_window = min(trend_window, len(trend) - 1)
            sg_window = sg_window if sg_window % 2 == 1 else sg_window - 1
            
            if sg_window >= 3:
                # 应用滤波器
                smoothed_trend = savgol_filter(trend, sg_window, 3)
                
                # 调整季节性以保持重建一致性
                adjusted_seasonal = seasonal + (trend - smoothed_trend)
                
                # 更新结果
                trend = smoothed_trend
                seasonal = adjusted_seasonal
        except Exception as e:
            print(f"Savgol平滑失败: {e}")
        
        return trend, seasonal
    def _generate_online_features(self, batch_x):
        """
        基于输入批次(batch_x)在线生成趋势和季节性特征
        使用与原始特征提取脚本完全相同的方法和参数
        
        参数:
        batch_x: 形状为[batch_size, seq_len, feature_dim]的tensor，其中feature_dim≥ORIGINAL_FEATURES
        
        返回:
        形状为[batch_size, seq_len, ORIGINAL_FEATURES*2]的tensor，包含生成的趋势和季节性特征
        """
        # 将tensor转为numpy进行处理(只使用原始特征)
        batch_numpy = batch_x[:, :, :ORIGINAL_FEATURES].detach().cpu().numpy()
        batch_size, seq_len, feature_dim = batch_numpy.shape
        
        # 初始化趋势和季节性特征数组
        ts_features = np.zeros((batch_size, seq_len, ORIGINAL_FEATURES * 2))
        
        # 处理每个样本
        for b in range(batch_size):
            # 处理每个原始特征
            # if b % 1 == 0:
            #     print(f"处理样本 {b+1}/{batch_size}")
            for f in range(ORIGINAL_FEATURES):
                # 获取当前特征的时间序列
                ts = batch_numpy[b, :, f]
                
                # 对特征使用滑动窗口分解方法
                trend, seasonal = self._sliding_window_decomposition_for_single_series(ts, period=24)
                
                # 存储分解结果（交错排列趋势和季节性）
                ts_features[b, :, f*2] = trend
                ts_features[b, :, f*2+1] = seasonal
        
        # 将numpy数组转回tensor并放到正确的设备上
        return torch.tensor(ts_features, dtype=torch.float32).to(self.device)

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y)
                # 验证始终只评估原始特征
                pred_eval, true_eval = self._get_features_for_evaluation(pred, true)
                loss = criterion(pred_eval.detach().cpu(), true_eval.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        # scale_statistic = {'mean': train_data.scaler.mean_, 'std': train_data.scaler.std}
        # with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
        #     pickle.dump(scale_statistic, f)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y)
                # 根据设置决定训练时使用哪些特征计算损失
                pred_train, true_train = self._get_features_for_training(pred, true)
                loss = criterion(pred_train, true_train)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')
        
        return self.model

    def test(self, setting, save_pred=False, inverse=False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        # 如果是在线生成特征且需要保存结果
        online_features_buffer = [] if (self.feature_mode == 'online' and self.save_online_features) else None
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                if i % 20 == 0:
                    print(f"Testing batch {i}/{len(test_loader)}")
                
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse, online_features_buffer)
                
                # 评估始终只关注原始特征
                pred_eval, true_eval = self._get_features_for_evaluation(pred, true)
                
                batch_size = pred_eval.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(
                    pred_eval.detach().cpu().numpy(), 
                    true_eval.detach().cpu().numpy()
                )) * batch_size
                metrics_all.append(batch_metric)
                
                if save_pred:
                    preds.append(pred_eval.detach().cpu().numpy())
                    trues.append(true_eval.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
        
        # 如果需要，保存在线生成的特征
        if online_features_buffer is not None and len(online_features_buffer) > 0:
            online_features = np.concatenate(online_features_buffer, axis=0)
            np.save(folder_path+'online_features.npy', online_features)
            print(f"保存在线生成的特征到 {folder_path}online_features.npy, 形状: {online_features.shape}")

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False, online_features_buffer=None):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        # 如果处于评估模式且选择了在线特征生成
        # print(not self.model.train)
        if self.model.eval and self.feature_mode == 'online':
            # print(self.feature_mode)
            # 生成在线特征
            extra_features = self._generate_online_features(batch_x[:, :, :ORIGINAL_FEATURES])
            
            # 如果需要保存在线特征
            if online_features_buffer is not None:
                online_features_buffer.append(extra_features.detach().cpu().numpy())
            
            # 结合原始特征和生成的特征
            # 判断原始特征维度
            _, _, feature_dim = batch_x.shape
            
            # 检查是否有足够的特征列
            has_extra_features = feature_dim > ORIGINAL_FEATURES
            has_time_features = feature_dim > ORIGINAL_FEATURES + ORIGINAL_FEATURES * 2
            
            if has_extra_features:
                # 提取原始特征部分和时间特征部分（如果有）
                orig_features = batch_x[:, :, :ORIGINAL_FEATURES]
                
                if has_time_features:
                    time_features = batch_x[:, :, ORIGINAL_FEATURES + ORIGINAL_FEATURES*2:]
                    # 重组: 原始特征 + 在线生成的特征 + 时间特征
                    batch_x = torch.cat([orig_features, extra_features, time_features], dim=2)
                else:
                    # 重组: 原始特征 + 在线生成的特征
                    batch_x = torch.cat([orig_features, extra_features], dim=2)
            else:
                # 只有原始特征，添加生成的特征
                batch_x = torch.cat([batch_x, extra_features], dim=2)

        outputs = self.model(batch_x)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y
    
    def eval(self, setting, save_pred=False, inverse=False):
        """evaluate a saved model"""
        args = self.args
        
        # ETT数据集分割 [12*30*24, 4*30*24, 4*30*24] train,val,test 
        data_split = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 4*30*24 + 4*30*24]
        
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len],  
            data_split=data_split,
            scale=True,
            scale_statistic=args.scale_statistic,
            use_precomputed_features=(self.feature_mode == 'precomputed')
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        # 如果是在线生成特征且需要保存结果
        online_features_buffer = [] if (self.feature_mode == 'online' and self.save_online_features) else None
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(data_loader):
                if i % 1 == 0:
                    print(f"Evaluating batch {i}/{len(data_loader)}")
                
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse, online_features_buffer)
                
                # 评估始终只关注原始特征
                pred_eval, true_eval = self._get_features_for_evaluation(pred, true)
                
                batch_size = pred_eval.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(
                    pred_eval.detach().cpu().numpy(), 
                    true_eval.detach().cpu().numpy()
                )) * batch_size
                metrics_all.append(batch_metric)
                
                if save_pred:
                    preds.append(pred_eval.detach().cpu().numpy())
                    trues.append(true_eval.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
            
        # 如果需要，保存在线生成的特征
        if online_features_buffer is not None and len(online_features_buffer) > 0:
            online_features = np.concatenate(online_features_buffer, axis=0)
            np.save(folder_path+'online_features.npy', online_features)
            print(f"保存在线生成的特征到 {folder_path}online_features.npy, 形状: {online_features.shape}")

        return mae, mse, rmse, mape, mspe