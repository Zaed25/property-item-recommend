import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
#  CUSTOM TRANSFORMER CLASS DEFINITIONS (CORRECTED VERSION)
# =============================================================================

class InitialCleaner(BaseEstimator, TransformerMixin):
    """Performs initial data cleaning by setting invalid values to NaN."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        print("-> Running InitialCleaner...")
        X_copy = X.copy()
        for col in ['bed', 'bath']:
            if col in X_copy.columns:
                mask = (X_copy[col] < 0) | (X_copy[col] > 20)
                X_copy.loc[mask, col] = np.nan
        if 'price' in X_copy.columns: X_copy.loc[X_copy['price'] < 1000, 'price'] = np.nan
        if 'house_size' in X_copy.columns: X_copy.loc[(X_copy['house_size'] < 100) | (X_copy['house_size'] > 15000), 'house_size'] = np.nan
        if 'acre_lot' in X_copy.columns: X_copy.loc[(X_copy['acre_lot'] < 0.01) | (X_copy['acre_lot'] > 1000), 'acre_lot'] = np.nan
        return X_copy

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates new features and drops columns that are no longer needed."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        print("-> Engineering new features...")
        X_copy = X.copy()
        if 'prev_sold_date' in X_copy.columns:
            X_copy['has_prev_sale'] = X_copy['prev_sold_date'].notna()
        else:
            X_copy['has_prev_sale'] = False
        price_bins = [0, 100_000, 300_000, 600_000, 1_000_000, float('inf')]
        price_labels = ['<100k', '100k-300k', '300k-600k', '600k-1M', '1M+']
        X_copy['price_range'] = pd.cut(X_copy['price'], bins=price_bins, labels=price_labels)
        return X_copy.drop(columns=['prev_sold_date', 'street'], errors='ignore')

class LogAndCapTransformer(BaseEstimator, TransformerMixin):
    """Applies a log transform and then caps outliers. Handles array input."""
    def __init__(self):
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) # Convert to DataFrame to be safe
        X_log = np.log1p(X_df)
        for i in range(X_df.shape[1]): # Iterate by column index
            col = X_log.iloc[:, i]
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[i] = Q1 - 1.5 * IQR
            self.upper_bounds_[i] = Q3 + 1.5 * IQR
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) # Convert to DataFrame
        X_log = np.log1p(X_df)
        for i in range(X_df.shape[1]):
             X_log.iloc[:, i] = X_log.iloc[:, i].clip(self.lower_bounds_[i], self.upper_bounds_[i])
        return X_log.to_numpy() # Return an array as is standard

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features with their frequencies. Handles array input."""
    def __init__(self): self.freq_map_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) # Convert to DataFrame
        for i in range(X_df.shape[1]):
            self.freq_map_[i] = X_df.iloc[:, i].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) # Convert to DataFrame
        X_copy = X_df.copy()
        for i in range(X_df.shape[1]):
            X_copy[f'col_{i}_encoded'] = X_copy.iloc[:, i].map(self.freq_map_[i]).fillna(0)
        return X_copy[[f'col_{i}_encoded' for i in range(X_df.shape[1])]].to_numpy()
