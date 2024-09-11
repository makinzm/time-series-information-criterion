"""dataを用意するmodule."""
from typing import Sequence

import numpy as np

def generate_ar5_data(n: int, coeffs: Sequence, noise_std: float) -> np.array:
    """AR(5)のデータを生成する関数.

    Args:
        n (int): データ数.
        coeffs (Sequence): AR(5)の係数.
        noise_std (float): ノイズの標準偏差.

    Returns:
        np.array: AR(5)のデータ.
    """
    coeffs = np.array(coeffs)
    p = len(coeffs)
    data = np.zeros(n + p)
    data[:p] = np.random.normal(size=p)
    for i in range(p, n + p):
        data[i] = np.dot(data[i - p:i], coeffs) + np.random.normal(scale=noise_std)
    return data[p:]

