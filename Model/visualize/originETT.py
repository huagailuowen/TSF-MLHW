import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置样式
# plt.style.use('seaborn-darkgrid')  # 推荐替代风格
sns.set_context("notebook")

# 读取数据
df = pd.read_csv('/mnt/d/周宸源/大学/学习/ML/TSF/Model/datasets/ETTh1.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# 提取变量
time = df['date']
features = df.columns[1:]  # 所有除 'date' 外的列

# 创建子图
num_features = len(features)
cols = 2
rows = (num_features + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharex=True)

axes = axes.flatten()  # 展平以便索引
for i, col in enumerate(features):
    axes[i].plot(time, df[col], label=col, linewidth=0.8)
    axes[i].set_title(col)
    axes[i].grid(True)

# 隐藏多余的子图框
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("ETTh1 原始特征时间序列", fontsize=16, y=1.02)
plt.show()