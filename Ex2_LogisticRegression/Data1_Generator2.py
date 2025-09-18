import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 生成数据 - 使用兼容旧版本的参数
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=2,  # 每类多个簇增加分散度
    weights=[0.85, 0.15],
    class_sep=1.0,           # 中等分离度
    flip_y=0.1,              # 适量噪声
    random_state=42
)

# 通过后期处理增加分散度
X = X * 2.5  # 手动扩大数据分布范围

# 扩大特征范围并增加变化
X[:, 0] = np.round(X[:, 0] * 40 + 100).clip(60, 220)  # 血糖(60-220 mg/dL)
X[:, 1] = np.round(X[:, 1] * 8 + 25).clip(15, 45)     # BMI(15-45 kg/m²)

# 创建DataFrame
df = pd.DataFrame({
    'Glucose': X[:, 0],
    'BMI': X[:, 1],
    'Outcome': y
})

# 更细致的业务逻辑调整
def adjust_outcome(row):
    glucose, bmi = row['Glucose'], row['BMI']
    
    # 极高风险区域
    if glucose > 200 and bmi > 38:
        return 1
    # 高风险区域
    elif glucose > 180 and bmi > 35:
        return 1 if np.random.random() > 0.2 else 0  # 80%概率为1
    # 低风险区域
    elif glucose < 90 and bmi < 22:
        return 0 if np.random.random() > 0.2 else 1  # 80%概率为0
    # 极低风险区域
    elif glucose < 80 and bmi < 20:
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
            cmap='viridis', edgecolors='k', s=40)
plt.xlabel('Glucose (mg/dL)', fontsize=12)
plt.ylabel('BMI (kg/m²)', fontsize=12)
plt.title('糖尿病数据集分布 (蓝色=健康, 黄色=患病)', fontsize=14)
plt.colorbar(label='Outcome (0=健康, 1=患病)')
plt.grid(True, alpha=0.3)
plt.show()

# 保存为CSV
df.to_csv('Ex2Data1_sample.csv', index=False)