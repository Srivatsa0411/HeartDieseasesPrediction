import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def clean_data(df, train_ref=None):
    for col in ['Cholesterol', 'RestingBP']:
        df[col] = df[col].apply(lambda x: np.nan if x <= 0 else x)
    
    for col in ['Cholesterol', 'RestingBP']:
        upper = df[col].quantile(0.99)
        df[col] = np.where(df[col] > upper, upper, df[col])
    
    median_ref = train_ref if train_ref is not None else df
    for col in ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
        df[col].fillna(median_ref[col].median(), inplace=True)
    
    for col in CATEGORICAL_FEATURES:
        df[col].fillna(median_ref[col].mode()[0], inplace=True)
    
    return df

def build_preprocessor():
    return ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), CATEGORICAL_FEATURES)
    ])