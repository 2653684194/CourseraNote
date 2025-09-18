import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 生成更均衡的数据
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=2,  # 每类多个簇增加分散度
    weights=[0.85, 0.15],
    class_sep=1.2,           # 适中的分离度
    flip_y=0.15,             # 适量噪声
    random_state=42
)

# 中心化并缩放数据
X = (X - X.mean(axis=0)) * 2.2  # 中心化后扩大分布范围

# 转换为医学指标范围
X[:, 0] = np.round(X[:, 0] * 25 + 100).clip(70, 180)  # 血糖(70-180 mg/dL)
X[:, 1] = np.round(X[:, 1] * 5 + 25).clip(18, 38)     # BMI(18-38 kg/m²)

# 创建DataFrame
df = pd.DataFrame({
    'Glucose': X[:, 0],
    'BMI': X[:, 1],
    'Outcome': y
})

# 定义更合理的业务逻辑调整函数
def adjust_outcome(row):
    glucose, bmi = row['Glucose'], row['BMI']
    
    # 极高风险区域 - 确定性设为患病
    if glucose > 160 and bmi > 32:
        return 1
    # 高风险区域 - 高概率设为患病
    elif glucose > 140 and bmi > 30:
        return 1 if np.random.random() > 0.25 else 0  # 75%概率为1
    # 低风险区域 - 高概率设为健康
    elif glucose < 100 and bmi < 23:
        return 0 if np.random.random() > 0.25 else 1  # 75%概率为0
    # 极低风险区域 - 确定性设为健康
    elif glucose < 85 and bmi < 20:
        return 0
    else:
        return row['Outcome']

df['Outcome'] = df.apply(adjust_outcome, axis=1)

# 打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 查看分布
print("类别分布:")
print(df['Outcome'].value_counts(normalize=True))
print("\n糖尿病患病率：{:.1f}%".format(df['Outcome'].mean() * 100))

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(df['Glucose'], df['BMI'], c=df['Outcome'], alpha=0.6, 
            cmap='coolwarm', edgecolors='k', s=40)
plt.xlabel('Glucose (mg/dL)', fontsize=12)
plt.ylabel('BMI (kg/m²)', fontsize=12)
plt.title('改进版糖尿病数据集分布 (蓝色=健康, 红色=患病)', fontsize=14)
plt.colorbar(label='Outcome (0=健康, 1=患病)')
plt.grid(True, alpha=0.3)

# 添加参考线显示决策边界
plt.axvline(x=140, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=30, color='gray', linestyle='--', alpha=0.5)
plt.text(142, 18, '血糖临界值', color='gray')
plt.text(70, 31, 'BMI临界值', color='gray')

plt.show()

# 保存为CSV
df.to_csv('Ex2Data1_sample.csv', index=False)