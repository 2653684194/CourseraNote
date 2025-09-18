import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class LinearSVM:
    def __init__(self, C=1.0, learning_rate=0.01, max_iter=1000):
        self.C = C  # 惩罚参数
        self.learning_rate = learning_rate  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.w = None  # 权重向量
        self.b = 0  # 偏置项
        self.X_mean = None  # 训练数据均值
        self.X_std = None  # 训练数据标准差
        self.loss_history = []  # 损失历史记录
    
    def fit(self, X, y):
        """使用梯度下降训练线性SVM"""
        m, n = X.shape
        
        # 数据标准化（非常重要，避免数值问题）
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8  # 防止除以0
        X_scaled = (X - self.X_mean) / self.X_std
        
        # 初始化权重和偏置
        self.w = np.zeros(n)
        self.b = 0
        self.loss_history = []
        
        # 训练循环
        for i in range(self.max_iter):
            # 计算所有样本的预测得分
            scores = np.dot(X_scaled, self.w) + self.b
            
            # 计算hinge loss的梯度
            # 对于满足 y_i(w·x_i + b) >= 1 的样本，梯度只有正则化部分
            # 对于不满足条件的样本，梯度包括正则化部分和hinge loss部分
            margin = y * scores
            
            # 计算损失
            loss = 0.5 * np.dot(self.w, self.w) + self.C * np.sum(np.maximum(0, 1 - margin)) / m
            self.loss_history.append(loss)
            
            # 计算梯度
            dw = self.w.copy()
            db = 0
            
            # 找出违反间隔的样本
            violate_indices = np.where(margin < 1)[0]
            if len(violate_indices) > 0:
                # 对违反间隔的样本计算梯度
                dw -= self.C * np.dot(y[violate_indices], X_scaled[violate_indices]) / m
                db -= self.C * np.sum(y[violate_indices]) / m
            
            # 梯度下降更新参数
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # 每100次迭代打印一次损失
            if (i + 1) % 100 == 0 or i == 0:
                print(f'迭代 {i+1}/{self.max_iter}, 损失: {loss:.6f}')
        
        # 计算训练准确率
        train_accuracy = self.score(X, y)
        print(f'训练完成，训练准确率: {train_accuracy:.4f}')
        
        return self
    
    def predict(self, X):
        """预测新数据的类别"""
        # 对输入数据进行标准化
        X_scaled = (X - self.X_mean) / self.X_std
        # 计算得分并预测类别
        scores = np.dot(X_scaled, self.w) + self.b
        return np.sign(scores)
    
    def score(self, X, y):
        """计算模型在给定数据上的准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def plot_decision_boundary(self, X, y):
        """绘制决策边界和数据点"""
        plt.figure(figsize=(10, 8))
        
        # 绘制数据点
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='正类')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='x', label='负类')
        
        # 创建网格以绘制决策边界
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        # 预测网格点的类别
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界和决策区域
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        
        # 计算并标记支持向量（近似）
        # 支持向量是那些满足 |w·x + b| ≈ 1 的样本
        X_scaled = (X - self.X_mean) / self.X_std
        scores = np.dot(X_scaled, self.w) + self.b
        sv_indices = np.where(np.abs(scores) <= 1.1)[0]  # 允许一些容差
        
        if len(sv_indices) > 0:
            plt.scatter(X[sv_indices, 0], X[sv_indices, 1], s=150, 
                        facecolors='none', edgecolors='green', label='支持向量')
        
        plt.title(f'线性SVM决策边界 (C={self.C})')
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.title('训练过程中的损失变化')
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.grid(True)
        plt.show()

# 加载和预处理数据
import os

# 加载和预处理数据
def load_data():
    data = loadmat('d:/AAA_Jupyter/Couresa_ML/Ex6_SVM/data/ex6data1.mat')
    X = data['X']
    y = data['y'].flatten()
    
    # 将标签从0-1转换为-1-1
    y = np.where(y == 0, -1, 1)
    
    print(f'数据集信息: {X.shape[0]}个样本, {X.shape[1]}个特征')
    print(f'正类样本数: {np.sum(y == 1)}')
    print(f'负类样本数: {np.sum(y == -1)}')
    
    return X, y

# 主函数
def main():
    # 加载数据
    print('加载数据集...')
    X, y = load_data()
    
    # 分割数据集为训练集和测试集（简单分割）
    # 这里为了简单起见，我们使用全部数据进行训练和测试
    # 在实际应用中，应该使用交叉验证或单独的测试集
    
    # 训练模型并测试不同的C值
    print('\n=== 测试不同惩罚参数C的效果 ===')
    
    # 测试C=1.0
    print('\n训练模型 (C=1.0)...')
    start_time = time.time()
    svm_c1 = LinearSVM(C=1.0, learning_rate=0.01, max_iter=2000)
    svm_c1.fit(X, y)
    end_time = time.time()
    print(f'训练耗时: {end_time - start_time:.4f}秒')
    svm_c1.plot_decision_boundary(X, y)
    
    # 测试C=10.0（更严格的分类）
    print('\n训练模型 (C=10.0)...')
    start_time = time.time()
    svm_c10 = LinearSVM(C=10.0, learning_rate=0.01, max_iter=2000)
    svm_c10.fit(X, y)
    end_time = time.time()
    print(f'训练耗时: {end_time - start_time:.4f}秒')
    svm_c10.plot_decision_boundary(X, y)
    
    # 测试C=0.1（更宽松的分类）
    print('\n训练模型 (C=0.1)...')
    start_time = time.time()
    svm_c01 = LinearSVM(C=0.1, learning_rate=0.01, max_iter=2000)
    svm_c01.fit(X, y)
    end_time = time.time()
    print(f'训练耗时: {end_time - start_time:.4f}秒')
    svm_c01.plot_decision_boundary(X, y)

if __name__ == '__main__':
    main()