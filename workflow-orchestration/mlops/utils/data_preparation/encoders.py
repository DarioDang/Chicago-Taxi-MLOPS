from typing import Tuple, Optional
import pandas as pd
import scipy
from sklearn.feature_extraction import DictVectorizer

def encode_features(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None
) -> Tuple[scipy.sparse.csr_matrix, Optional[scipy.sparse.csr_matrix], DictVectorizer]:
    """
    Encode categorical and numerical features using DictVectorizer.
    Equivalent to pd.get_dummies(drop_first=True), but more efficient for sparse data.

    Returns:
        - Encoded X_train (sparse matrix)
        - Encoded X_val (sparse matrix or None)
        - The fitted DictVectorizer
    """
    dv = DictVectorizer(sparse=True)

    # Convert DataFrame to list of dicts
    train_dicts = X_train.to_dict(orient='records')
    X_train_enc = dv.fit_transform(train_dicts)

    X_val_enc = None
    if X_val is not None:
        val_dicts = X_val[X_train.columns].to_dict(orient='records')
        X_val_enc = dv.transform(val_dicts)

    return X_train_enc, X_val_enc, dv
