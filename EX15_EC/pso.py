import numpy as np


class Problem:
    def __init__(self, func, dim, lb, ub):
        self.func: callable = func
        self.dim: int = dim
        self.lb: np.ndarray = lb if isinstance(lb, np.ndarray) else np.full((dim,), lb) # lower bound: scalar or vector
        self.ub: np.ndarray = ub if isinstance(ub, np.ndarray) else np.full((dim,), ub) # upper bound: scalar or vector

    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            return self.func(x)
        elif x.ndim == 2:
            return np.apply_along_axis(self.func, 1, x)
        else:
            return self.func(x)
    def __getitem__(self):
        return self.lb, self.ub, self.dim
class PSO:
    def __init__(self, tar_pro: Problem, pop_size, max_iter, w, c1, c2, weight_decay=0.999):
        self.tar_pro = tar_pro 
        self.pop_size = pop_size # population size
        self.max_iter = max_iter # maximum iteration number
        self.w = w # inertia weight
        self.c1 = c1 # social learning constant: 0-1
        self.c2 = c2 # cognitive learning constant: 0-1
        self.weight_decay = weight_decay # weight decay factor

        self.part_pos = self.random_init() # particle position
        self.part_vel = self.random_init() # particle velocity
        self.part_best_pos = self.part_pos.copy() 
        self.global_best_pos = np.random.rand(self.tar_pro.dim)

    def random_init(self, scale=1.0):
        return np.random.uniform(self.tar_pro.lb, self.tar_pro.ub, size=(self.pop_size, self.tar_pro.dim)) * scale
    
    def latin_hypercube_init(self, scale=1.0):
        """
        拉丁超立方初始化：在每个维度上分层采样，保证更均匀的覆盖
        """
        positions = np.zeros((self.pop_size, self.tar_pro.dim))
        
        for j in range(self.tar_pro.dim):
            # 将区间分成 pop_size 层
            segments = np.linspace(self.tar_pro.lb[j], self.tar_pro.ub[j], self.pop_size + 1)
            
            # 每层随机取一个点
            for i in range(self.pop_size):
                positions[i, j] = np.random.uniform(segments[i], segments[i+1])
            
            # 打乱顺序，避免维度间的相关性
            np.random.shuffle(positions[:, j])
        
        return positions * scale

    def optimize(self):
        """
        优化函数
        """
        for _ in range(self.max_iter):
            # 更新粒子速度（添加随机因子）
            r1 = np.random.rand(self.pop_size, self.tar_pro.dim)
            r2 = np.random.rand(self.pop_size, self.tar_pro.dim)
            self.part_vel = (self.w * self.part_vel + 
                           self.c1 * r1 * (self.part_best_pos - self.part_pos) + 
                           self.c2 * r2 * (self.global_best_pos - self.part_pos))
            
            # 更新粒子位置
            self.part_pos = self.part_pos + self.part_vel
            self.part_pos = np.clip(self.part_pos, self.tar_pro.lb, self.tar_pro.ub)
            
            # 更新粒子最佳位置
            current_fitness = self.tar_pro(self.part_pos)
            best_fitness = self.tar_pro(self.part_best_pos)
            self.part_best_pos = np.where(current_fitness[:, np.newaxis] < best_fitness[:, np.newaxis], self.part_pos, self.part_best_pos)
            
            # 更新全局最佳位置
            self.global_best_pos = self.part_best_pos[np.argmin(self.tar_pro(self.part_best_pos))]
            
            if _ % 10 == 0:
                self.w *= self.weight_decay
                self.c1 *= self.weight_decay
                self.c2 *= self.weight_decay
            
            if _ % 100 == 0:
                self.weight_decay *= self.weight_decay

                print(self.tar_pro(self.global_best_pos))
        
        return self.global_best_pos
    
