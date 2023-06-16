
# DON'T NEED IT, signal already preprocessed 

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.covariance import EllipticEnvelope

"""
    Preprocesses the data by performing normalization, data cleaning,
    and removal of any outliers using specific techniques to prepare the data for training the CVAE model.

    Parameters:
    - X: numerical array-like; the features/data to be preprocessed.
    - y: numerical array-like; the corresponding labels/targets.

    Returns:
    - X_processed: numerical array-like; the preprocessed features.
    - y_processed: numerical array-like; the preprocessed labels.

"""

def preprocess_data(X, y):
    # Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Data cleaning
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Removal of any outliers
    outlier_detector = EllipticEnvelope(contamination=0.1)
    outlier_detector.fit(X)
    outlier_mask = outlier_detector.predict(X) != -1
    X = X[outlier_mask]
    y = y[outlier_mask]

    return X, y
