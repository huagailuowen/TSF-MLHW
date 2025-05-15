import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'  # 避免中文乱码
plt.rcParams['figure.figsize'] = (14, 8)

# 读取数据
df = pd.read_csv('/mnt/d/周宸源/大学/学习/ML/TSF/Model/datasets/ETTh1.csv', parse_dates=['date'])

# 筛选前 3 天
samples_per_day = 24
days = 7*4
subset = df.iloc[:samples_per_day * days]

# 绘制多个子图
cols = df.columns[1:]  # 除 date 外的列
num_cols = len(cols)

fig, axes = plt.subplots(num_cols, 1, figsize=(14, 3 * num_cols), sharex=True)

for i, col in enumerate(cols):
    ax = axes[i]
    ax.plot(subset['date'], subset[col], label=col)
    ax.set_title(f"{col} - 前 {days} 天")
    ax.legend(loc='upper right')
    ax.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
