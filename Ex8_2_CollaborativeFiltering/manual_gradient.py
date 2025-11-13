import numpy as np

class Variable:
    """\支持自动梯度计算的变量类"""
    def __init__(self, data, name=None, requires_grad=True):
        # 确保数据是numpy数组
        self.data = np.array(data, dtype=np.float64) if not isinstance(data, np.ndarray) else data
        self.name = name
        self.requires_grad = requires_grad
        
        # 梯度初始化为None
        self.grad = None
        
        # 记录操作历史，用于反向传播
        self._prev = set()
        self._op = None  # 生成该变量的操作
        
    def __repr__(self):
        name_str = f"{self.name}: " if self.name else ""
        requires_grad_str = "(requires_grad=True)" if self.requires_grad else ""
        return f"Variable({name_str}{self.data.shape}{requires_grad_str})"
    
    def backward(self, grad_output=None):
        """执行反向传播计算梯度"""
        # 如果没有梯度或者不需要计算梯度，则直接返回
        if not self.requires_grad:
            return
        
        # 初始化梯度，默认是单位矩阵
        if grad_output is None:
            if self.data.ndim == 0:  # 标量
                grad_output = 1.0
            else:  # 张量，初始化为单位矩阵
                grad_output = np.ones_like(self.data)
        
        # 累加梯度
        if self.grad is None:
            self.grad = np.array(grad_output)
        else:
            self.grad += np.array(grad_output)
        
        # 递归计算前驱节点的梯度
        if self._op is not None:
            self._op.backward(grad_output)

class Operation:
    """基本操作的基类"""
    def forward(self):
        """前向传播"""
        pass
    
    def backward(self, grad_output):
        """反向传播"""
        pass

class Add(Operation):
    """加法操作"""
    def __init__(self, a, b):
        self.a = a
        self.b = b
        # 执行前向传播
        self.output = Variable(self.forward(), requires_grad=a.requires_grad or b.requires_grad)
        # 记录操作历史
        self.output._prev = {a, b}
        self.output._op = self
    
    def forward(self):
        return self.a.data + self.b.data
    
    def backward(self, grad_output):
        # 加法的梯度是梯度直接传递给两个操作数
        if self.a.requires_grad:
            self.a.backward(grad_output)
        if self.b.requires_grad:
            self.b.backward(grad_output)

class Matmul(Operation):
    """矩阵乘法操作"""
    def __init__(self, a, b):
        self.a = a
        self.b = b
        # 执行前向传播
        self.output = Variable(self.forward(), requires_grad=a.requires_grad or b.requires_grad)
        # 记录操作历史
        self.output._prev = {a, b}
        self.output._op = self
    
    def forward(self):
        return np.matmul(self.a.data, self.b.data)
    
    def backward(self, grad_output):
        # 矩阵乘法的梯度计算
        if self.a.requires_grad:
            # dL/da = grad_output * b^T
            grad_a = np.matmul(grad_output, self.b.data.T)
            self.a.backward(grad_a)
        if self.b.requires_grad:
            # dL/db = a^T * grad_output
            grad_b = np.matmul(self.a.data.T, grad_output)
            self.b.backward(grad_b)

class Transpose(Operation):
    """矩阵转置操作"""
    def __init__(self, a):
        self.a = a
        # 执行前向传播
        self.output = Variable(self.forward(), requires_grad=a.requires_grad)
        # 记录操作历史
        self.output._prev = {a}
        self.output._op = self
    
    def forward(self):
        return self.a.data.T
    
    def backward(self, grad_output):
        # 转置的梯度也是转置
        if self.a.requires_grad:
            self.a.backward(grad_output.T)

class Sum(Operation):
    """求和操作"""
    def __init__(self, a, axis=None, keepdims=False):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
        # 执行前向传播
        self.output = Variable(self.forward(), requires_grad=a.requires_grad)
        # 记录操作历史
        self.output._prev = {a}
        self.output._op = self
    
    def forward(self):
        return np.sum(self.a.data, axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, grad_output):
        # 求和操作的梯度需要广播回原始形状
        if self.a.requires_grad:
            if self.axis is None:
                # 标量求和，梯度是全1的矩阵
                grad_a = np.ones_like(self.a.data) * grad_output
            else:
                # 沿着特定轴求和，梯度需要广播
                shape = [1] * self.a.data.ndim
                if isinstance(self.axis, int):
                    shape[self.axis] = self.a.data.shape[self.axis]
                else:
                    for ax in self.axis:
                        shape[ax] = self.a.data.shape[ax]
                grad_a = np.reshape(grad_output, shape) * np.ones_like(self.a.data)
            self.a.backward(grad_a)

class Square(Operation):
    """平方操作"""
    def __init__(self, a):
        self.a = a
        # 执行前向传播
        self.output = Variable(self.forward(), requires_grad=a.requires_grad)
        # 记录操作历史
        self.output._prev = {a}
        self.output._op = self
    
    def forward(self):
        return np.square(self.a.data)
    
    def backward(self, grad_output):
        # 平方操作的梯度是 2 * a * grad_output
        if self.a.requires_grad:
            grad_a = 2 * self.a.data * grad_output
            self.a.backward(grad_a)

class GradientDescent:
    """梯度下降优化器"""
    def __init__(self, variables, learning_rate=0.01):
        self.variables = variables
        self.learning_rate = learning_rate
    
    def step(self):
        """执行一步梯度下降"""
        for var in self.variables:
            if var.requires_grad and var.grad is not None:
                var.data -= self.learning_rate * var.grad
    
    def zero_grad(self):
        """清零所有变量的梯度"""
        for var in self.variables:
            if var.requires_grad:
                var.grad = None

# 便捷函数用于创建操作

def add(a, b):
    """创建加法操作"""
    return Add(a, b).output

def matmul(a, b):
    """创建矩阵乘法操作"""
    return Matmul(a, b).output

def transpose(a):
    """创建矩阵转置操作"""
    return Transpose(a).output

def sum(a, axis=None, keepdims=False):
    """创建求和操作"""
    return Sum(a, axis=axis, keepdims=keepdims).output

def square(a):
    """创建平方操作"""
    return Square(a).output

def mean(a, axis=None, keepdims=False):
    """创建均值操作"""
    total = sum(a, axis=axis, keepdims=keepdims)
    if axis is None:
        count = np.prod(a.data.shape)
    elif isinstance(axis, int):
        count = a.data.shape[axis]
    else:
        count = np.prod([a.data.shape[ax] for ax in axis])
    return Variable(total.data / count, requires_grad=a.requires_grad)

# 示例：使用手动梯度计算实现线性回归
if __name__ == '__main__':
    # 创建一些测试数据
    X_data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y_data = np.array([[3.0], [5.0], [7.0], [9.0]])  # 目标: y = 1*X1 + 1*X2 + 0
    
    # 创建变量
    X = Variable(X_data, name='X', requires_grad=False)
    y = Variable(y_data, name='y', requires_grad=False)
    
    # 创建模型参数
    W = Variable(np.random.normal(0, 0.1, (2, 1)), name='W', requires_grad=True)
    b = Variable(np.zeros((1, 1)), name='b', requires_grad=True)
    
    # 创建优化器
    optimizer = GradientDescent([W, b], learning_rate=0.01)
    
    # 训练模型
    for epoch in range(1000):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        y_pred = add(matmul(X, W), b)
        loss = mean(square(add(y_pred, Variable(-y.data, requires_grad=False))))
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 打印进度
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.data:.6f}, W: {W.data.flatten()}, b: {b.data[0,0]:.6f}')
    
    print('\n训练完成!')
    print(f'最终权重 W: {W.data}')
    print(f'最终偏置 b: {b.data}')
    print(f'最终预测: {add(matmul(X, W), b).data}')