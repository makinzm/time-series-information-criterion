"""シードを整えるなどの、ユーティリティ関数を提供するmodule."""

import numpy as np

def set_seed(seed: int) -> None:
    """シードを設定する関数.

    Args:
        seed (int): シードの値.

    Returns:
        None
    """
    np.random.seed(seed)

