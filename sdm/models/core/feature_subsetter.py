from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSubsetter(BaseEstimator, TransformerMixin):
    """
    Custom transformer to subset a DataFrame to a specified list of feature columns.
    Stores the feature names as a class attribute for later reference.
    Compatible with scikit-learn pipelines and pickling.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # No fitting necessary, just return self
        return self

    def transform(self, X):
        # Subset and order columns
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        else:
            # If X is not a DataFrame, try to convert it
            return pd.DataFrame(X, columns=self.feature_names)[self.feature_names] 