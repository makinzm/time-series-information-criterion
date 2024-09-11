"""モデルを作成するためのmodule."""

import pymc as pm

# TODO(makinzm): I think priors should be a class, not a dict to make type hinting easier.
def create_ar_model(n_chains: int, priors: dict, n_steps: int, data_len: int) -> pm.Model:
    """
    ARモデルを作成する関数.

    Args:
        n_chains (int): MCMCのチェーン数.
        priors (dict): 事前分布.
            coefs: 係数の事前分布(uniform | normal).
                distribution: uniform
                    lower: 係数の下限.
                    upper: 係数の上限.
                distribution: normal
                    mu: 係数の平均.
                    sigma: 係数の標準偏差.
            sigma: ノイズの事前分布の標準偏差.
        n_steps (int): MCMCのステップ数.
        data_len (int): データの長さ.

    Returns:
        pymc.Model: ARモデル.
    """
    model = pm.Model()
    with model:
        coefs = None
        if priors["coefs"]["distribution"] == "uniform":
            coefs = pm.Uniform("coefs", lower=priors["coefs"]["lower"], upper=priors["coefs"]["upper"], shape=n_chains + 1)
        elif priors["coefs"]["distribution"] == "normal":
            coefs = pm.Normal("coefs", mu=priors["coefs"]["mu"], sigma=priors["coefs"]["sigma"], shape=n_chains + 1)
        sigma = pm.HalfNormal("sigma", sigma=priors["sigma"])
        data = pm.AR(f"AR({n_chains})", coefs, sigma, constant=True, steps=data_len - n_chains)

    return model

