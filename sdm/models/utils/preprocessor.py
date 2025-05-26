# Placeholder for data preprocessing functions (e.g., scaling, transformation)
# specifically for preparing data for model input.

import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer # Example & for ZeroDivScaler
import numpy as np # For add_epsilon
import logging

# Example function (to be replaced with actual logic from notebook)
def scale_features(data_df: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
    logging.info(f"Scaling features: {columns_to_scale}")
    # scaler = StandardScaler()
    # data_df_scaled = data_df.copy()
    # data_df_scaled[columns_to_scale] = scaler.fit_transform(data_df[columns_to_scale])
    # logging.info("Features scaled.")
    # return data_df_scaled
    return data_df # Placeholder


def add_epsilon_to_data(X: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Adds a small epsilon to the input data to avoid zero values, e.g., before log transform."""
    # Ensure X is a NumPy array for direct addition, or handle DataFrame case
    if isinstance(X, pd.DataFrame):
        X_mod = X.copy() + epsilon
    elif isinstance(X, np.ndarray):
        X_mod = X.copy() + epsilon
    else:
        # Attempt to convert to numpy array if it's some other array-like, or raise error
        try:
            X_mod = np.array(X, dtype=float) + epsilon
        except Exception as e:
            logging.error(f"Could not apply epsilon to data of type {type(X)}: {e}")
            raise TypeError(f"Input X for add_epsilon_to_data must be DataFrame, ndarray, or convertible. Got {type(X)}")
    return X_mod

class ZeroDivScaler(FunctionTransformer):
    """
    A custom scikit-learn compatible transformer that adds a small epsilon 
    to the data to prevent division by zero or issues with log(0) in subsequent transformations.
    Useful when data might contain true zeros but subsequent steps require positive values.
    """
    def __init__(self, epsilon: float = 1e-5, validate=False, accept_sparse=False, check_inverse=True, kw_args=None, inv_kw_args=None):
        """
        Args:
            epsilon (float): The small constant to add to the data.
        """
        self.epsilon = epsilon
        # Pass epsilon to the function through kw_args
        _kw_args = kw_args if kw_args is not None else {}
        _kw_args['epsilon'] = self.epsilon 
        super().__init__(
            func=add_epsilon_to_data, 
            inverse_func=None, # No inverse operation defined for simply adding epsilon
            validate=validate, 
            accept_sparse=accept_sparse, 
            check_inverse=check_inverse, 
            kw_args=_kw_args,
            inv_kw_args=inv_kw_args
        )

    # The fit method can often be a no-op if the transformation is stateless or state is handled in __init__
    # For FunctionTransformer, fit usually does nothing unless validate=True and it checks input data.
    # def fit(self, X, y=None):
    #     return super().fit(X, y)

    # transform is handled by the parent FunctionTransformer using the func provided
    # def transform(self, X):
    #     return super().transform(X) 