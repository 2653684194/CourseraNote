import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class RBFSVM:
    """
    支持RBF核的支持向量机(SVM)的底层实现
    使用梯度下降法训练模型，可以处理非线性分类问题
    
    RBF核原理：
    K(x, x') = exp(-γ||x - x'||²)
    其中γ是核函数的参数，控制核函数的宽度
    """
    def __init__(self, C=1.0, gamma=1.0, learning_rate=0.01, max_iter=1000):
        """
        初始化RBF SVM模型参数
        
        参数:
        - C: 惩罚参数，控制对错误分类的惩罚程度
        - gamma: RBF核参数，控制核函数的宽度
        - learning_rate: 学习率
        - max_iter: 最大迭代次数
        """
        self.C = C
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = None  # 拉格朗日乘子
        self.b = 0  # 偏置项
        self.X = None  # 训练数据
        self.y = None  # 训练标签
        self.loss_history = []  # 损失历史记录
    
    def _rbf_kernel(self, x1, x2):
        """计算RBF核函数"""
        if len(x1.shape) == 1 and len(x2.shape) == 1:
            # 单个样本与单个样本
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif len(x1.shape) == 1 and len(x2.shape) == 2:
            # 单个样本与多个样本
            return np.exp(-self.gamma * np.linalg.norm(x2 - x1, axis=1) ** 2)
        else:
            # 多个样本与多个样本
            pairwise_dists = np.linalg.norm(x1[:, np.newaxis] - x2, axis=2) ** 2
            return np.exp(-self.gamma * pairwise_dists)
    
    def fit(self, X, y):
        """
        训练RBF核SVM模型
        使用简化的SMO算法思想，但用梯度下降实现
        """
        m = X.shape[0]  # 样本数量
        
        # 保存训练数据
        self.X = X
        self.y = y
        
        # 初始化拉格朗日乘子
        self.alpha = np.zeros(m)
        self.b = 0
        self.loss_history = []
        
        # 计算核矩阵以提高效率
        K = self._rbf_kernel(X, X)
        
        # 训练循环
        for i in range(self.max_iter):
            # 计算预测值
            f = np.dot(self.alpha * y, K) + self.b
            
            # 计算损失值（近似）
            loss = 0.5 * np.dot(self.alpha * y, np.dot(K, self.alpha * y)) - np.sum(self.alpha)
            self.loss_history.append(loss)
            
            # 计算梯度
            grad_alpha = np.dot(K, self.alpha * y) * y - 1
            
            # 对alpha应用投影梯度下降
            # 先进行无约束更新
            self.alpha -= self.learning_rate * grad_alpha
            
            # 然后将alpha投影到[0, C]区间
            self.alpha = np.clip(self.alpha, 0, self.C)
            
            # 更新偏置b
            # 选择alpha在(0, C)之间的样本计算b
            sv_indices = np.where((self.alpha > 1e-5) & (self.alpha < self.C - 1e-5))[0]
            if len(sv_indices) > 0:
                self.b = np.mean(y[sv_indices] - np.dot(self.alpha * y, K[:, sv_indices]))
            
            # 每100次迭代打印一次损失
            if (i + 1) % 100 == 0 or i == 0:
                print(f'迭代 {i+1}/{self.max_iter}, 损失: {loss:.6f}')
        
        # 计算训练准确率
        y_pred = self.predict(X)
        train_accuracy = np.mean(y_pred == y)
        print(f'训练完成，训练准确率: {train_accuracy:.4f}')
        
        # 找出支持向量
        self.sv_indices = np.where(self.alpha > 1e-5)[0]
        print(f'支持向量数量: {len(self.sv_indices)} / {m}')
        
        return self
    
    def predict(self, X):
        """预测新数据的类别"""
        # 计算测试样本与所有训练样本的核函数值
        K = self._rbf_kernel(X, self.X)
        # 计算预测得分
        scores = np.dot(self.alpha * self.y, K.T) + self.b
        # 返回预测类别
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
        
        # 标记支持向量
        if hasattr(self, 'sv_indices'):
            plt.scatter(X[self.sv_indices, 0], X[self.sv_indices, 1], s=150, 
                        facecolors='none', edgecolors='green', label='支持向量')
        
        # 创建网格以绘制决策边界
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # 预测网格点的类别
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界和决策区域
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        
        plt.title(f'RBF核SVM决策边界 (C={self.C}, gamma={self.gamma})')
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
def load_ex6data2():
    """
    加载ex6data2.mat数据集
    """
    data = loadmat('d:/AAA_Jupyter/Couresa_ML/Ex6_SVM/data/ex6data2.mat')
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
    """
    主函数：加载ex6data2数据集，训练RBF核SVM模型，
    并可视化决策边界
    """
    # 加载数据
    print('加载ex6data2数据集...')
    X, y = load_ex6data2()
    
    # 分割数据集为训练集和测试集（简单分割）
    # 由于数据集较小，我们使用全部数据进行训练和测试
    
    # 训练RBF核SVM模型
    print('\n开始训练RBF核SVM模型...')
    start_time = time.time()
    
    # 选择合适的参数
    # 对于ex6data2，gamma值通常需要设置得较大一些
    svm = RBFSVM(C=1.0, gamma=20.0, learning_rate=0.001, max_iter=3000)
    svm.fit(X, y)
    
    end_time = time.time()
    print(f'SVM模型训练完成，耗时: {end_time - start_time:.4f}秒')
    
    # 评估模型
    accuracy = svm.score(X, y)
    print(f'模型准确率: {accuracy:.4f}')
    
    # 绘制决策边界
    print('\n绘制决策边界...')
    svm.plot_decision_boundary(X, y)
    
    # 尝试不同的gamma值
    print('\n使用不同的gamma值重新训练模型...')
    
    # gamma=5.0
    print('\n训练模型 (gamma=5.0)...')
    svm_g5 = RBFSVM(C=1.0, gamma=5.0, learning_rate=0.001, max_iter=3000)
    svm_g5.fit(X, y)
    svm_g5.plot_decision_boundary(X, y)
    
    # gamma=100.0
    print('\n训练模型 (gamma=100.0)...')
    svm_g100 = RBFSVM(C=1.0, gamma=100.0, learning_rate=0.001, max_iter=3000)
    svm_g100.fit(X, y)
    svm_g100.plot_decision_boundary(X, y)

if __name__ == '__main__':
    main()