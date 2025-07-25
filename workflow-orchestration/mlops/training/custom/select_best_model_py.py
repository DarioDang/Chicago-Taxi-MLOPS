from typing import Callable, Dict, List, Tuple, Union
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

@custom
def select_best_model(
    inputs: List[Tuple[
        Dict[str, Union[bool, float, int, str]],  # best_params
        csr_matrix,                               # X
        Series,                                   # y
        Dict[str, Union[Callable[..., BaseEstimator], str]]  # model_info
    ]],
    *args,
    **kwargs,
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series,
    Dict[str, Union[Callable[..., BaseEstimator], str]]
]:
    if not inputs:
        raise ValueError("No models were provided for selection.")

    # Initialize best model variables
    best_rmse = float("inf")
    best_model = None

    for result in inputs:
        params, X, y, info = result
        rmse = info.get("rmse", float("inf"))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = result

    if best_model is None:
        raise ValueError("No valid model found in input list.")

    return best_model
