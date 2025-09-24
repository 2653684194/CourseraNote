def PCA(X:np.ndarray,k:int)->tuple[np.ndarray,np.ndarray]:
    """
    X:(m,n)  m samples, n dimension
    """
    # 1.Standardization (or centering, we choose standardization here)
    m,n = X.shape
    if (k>n):
        return X
    
    mu = np.sum(X,axis=0)/m
    sigma = np.sqrt(np.sum(np.power(X-mu,2),axis=0)/m) # /m-1  可以获得无偏估计
    X_std = (X - mu)/sigma

    Cov = (X_std.T @ X_std)/m # (n,n)

    # find features values and vectors
    eigen_values,eigen_vects = np.linalg.eig(Cov)
    # fetch the largest k features values (meaning to reduce dimension to k)
    sorted_idx = np.sort(eigen_values).astype('int')[::-1]# [strat:end:step] 获得升序数组反转为降序
    print(-eigen_vects[sorted_idx])

    eigen_values = eigen_values[sorted_idx][:k]
    eigen_vects = eigen_vects[sorted_idx][:,:k] # (n,k)
    print(eigen_vects)

    # project data
    X_pca = X @ (eigen_vects) # (m,n)x(n,k)
    # X_pca = X_pca @ eigen_vects.T
    return X_pca,eigen_vects

# 简洁写法
def pca(X,k)->tuple[np.ndarray,np.ndarray]:
    X_std=(X-X.mean(axis=0))/X.std(axis=0)
    Cov = (X_std.T @ X_std)/X_std.shape[0]
    print(Cov.shape)
    
    P,A,P_T = np.linalg.svd(Cov) # 分解对角矩阵
    vects = P[:,:k]
    
    X = X @ vects
    
    # X = X @ vects.T
    return X , vects

