import numpy as np


def norm_1(A: np.ndarray) -> float:
    """
    1-范数（列和范数）
    max_j sum_i |a_ij|
    """
    A = np.asarray(A)
    return np.max(np.sum(np.abs(A), axis=0))


def norm_inf(A: np.ndarray) -> float:
    """
    ∞-范数（行和范数）
    max_i sum_j |a_ij|
    """
    A = np.asarray(A)
    return np.max(np.sum(np.abs(A), axis=1))


def norm_2(A: np.ndarray) -> float:
    """
    2-范数（谱范数）
    最大奇异值 σ_max
    """
    A = np.asarray(A)
    # 只需要最大奇异值，full_matrices=False 更快
    singular_values = np.linalg.svd(A, compute_uv=False)
    return singular_values[0]


def norm_fro(A: np.ndarray) -> float:
    """
    Frobenius 范数
    sqrt(sum_ij a_ij^2)
    """
    A = np.asarray(A)
    return np.sqrt(np.sum(A ** 2))


# def norm_nuclear(A: np.ndarray) -> float:
#     """
#     核范数（Nuclear norm）
#     奇异值之和 sum_i σ_i
#     """
#     A = np.asarray(A)
#     singular_values = np.linalg.svd(A, compute_uv=False)
#     return np.sum(singular_values)


# 可选：统一接口
def matrix_norm(A: np.ndarray, norm_type: str = "fro") -> float:
    """
    统一矩阵范数接口

    Parameters
    ----------
    A : np.ndarray
        输入矩阵
    norm_type : str
        可选:
        - "1"        : 1-范数
        - "inf"      : ∞-范数
        - "2"        : 2-范数（谱范数）
        - "fro"      : Frobenius 范数
        - "nuclear"  : 核范数
    """
    norm_type = norm_type.lower()

    if norm_type == "1":
        return norm_1(A)
    elif norm_type == "inf":
        return norm_inf(A)
    elif norm_type == "2":
        return norm_2(A)
    elif norm_type == "fro":
        return norm_fro(A)
    elif norm_type == "nuclear":
        return norm_nuclear(A)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    
def norm_nuclear(A: np.ndarray) -> float:
    A = np.asarray(A)
    if A.ndim == 1:
        # 向量的核范数 = L2 norm
        return float(np.linalg.norm(A, 2))
    return float(np.sum(np.linalg.svd(A, compute_uv=False)))



