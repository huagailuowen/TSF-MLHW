import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from tqdm import tqdm

# 读取数据
df = pd.read_csv('/mnt/d/周宸源/大学/学习/ML/TSF/Crossformer/datasets/ETTh1.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# 设置参数
target_cols = df.columns[1:]  # 所有时间序列变量（除 date 外）
forecast_horizon = 1          # 一步预测（可改为多步）
history_window = 7 * 96       # 使用过去 7 天数据（15min 采样 = 每天 96 条）

# 用于保存预测结果
result_df = df[['date']].copy()

# 预测每列
for col in tqdm(target_cols, desc="Processing Columns"):
    hw_preds = []
    arima_preds = []

    # 滑动窗口预测
    for i in range(history_window, len(df) - forecast_horizon):
        ts_window = df[col].iloc[i - history_window:i]

        # Holt-Winters
        try:
            hw_model = ExponentialSmoothing(ts_window, seasonal='add', seasonal_periods=96).fit()
            hw_forecast = hw_model.forecast(forecast_horizon)[-1]
        except:
            hw_forecast = np.nan

        # ARIMA
        try:
            arima_model = auto_arima(ts_window, seasonal=False, suppress_warnings=True, error_action="ignore")
            arima_forecast = arima_model.predict(n_periods=forecast_horizon)[-1]
        except:
            arima_forecast = np.nan

        hw_preds.append(hw_forecast)
        arima_preds.append(arima_forecast)

    # 用 NaN 补齐前面没有预测的时间点
    result_df[f'{col}_hw_pred'] = [np.nan] * history_window + hw_preds
    result_df[f'{col}_arima_pred'] = [np.nan] * history_window + arima_preds

# 保存为新的 CSV 文件
result_df.to_csv('/mnt/d/周宸源/大学/学习/ML/TSF/Crossformer/datasets/ETT_predictions_HW_ARIMA.csv', index=False)
print("预测特征文件已生成：ETT_predictions_HW_ARIMA.csv")
