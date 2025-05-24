import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler, normalize

class BaseProcessor:
    def __init__(self):
        self.iris_data = load_iris()
        self.cancer_data = load_breast_cancer()

    def get_iris_data(self):
        return self.iris_data.data, self.iris_data.target

    def get_cancer_data(self):
        return self.cancer_data.data, self.cancer_data.target

    def standard_scale(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def normalize_data(self, data, norm_type='l2'):
        return normalize(data, norm=norm_type)

    def calculate_correlation(self, X, y):
        return [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]

    def manual_standard_scaler(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / std
