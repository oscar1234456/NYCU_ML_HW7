import numpy as np
from scipy.spatial.distance import cdist


def get_kernel(X, kernel_option):
    # X: (135, 45045)
    if kernel_option == "no":
        print(f"kernel: no")
        # return covariance matrix
        S = np.cov(X, bias=True)
        # S = S/135
        return S
    elif kernel_option == "linear":
        print(f"kernel: linear")
        K = X @ X.T
    elif kernel_option == "polynomial":
        # polynomial (gamma * X @ X.T + coef) ** degree
        print(f"kernel: polynomial")
        gamma = 0.01
        coef = 0
        degree = 3
        K = (gamma * X @ X.T + coef) ** degree
    elif kernel_option == "RBF":
        print(f"kernel: RBF")
        # RBF: exp(-gamma * L2 square norm)
        gamma = 0.01
        K = np.exp(-gamma * cdist(X, X, metric='sqeuclidean'))
    else:
        print("not a correct kernel name")
        print("using 'no' option")
        # return covariance matrix
        S = np.cov(X, bias=True)
        return S

    # kernel PCA (assumed data are centered already)
    N = X.shape[0]  # the number of data
    one_N = np.ones((N, N)) / N
    K_C = K - one_N @ K - K @ one_N + one_N @ K @ one_N

    return K_C

