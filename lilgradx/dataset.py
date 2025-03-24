import pandas as pd
import numpy as np
from lilgradx.tensor import Tensor
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, file_path, target_column, drop_columns=[]):
        self.file_path = file_path
        self.target_column = target_column
        self.drop_columns = drop_columns
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_and_preprocess()

    def _load_and_preprocess(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=self.drop_columns, errors="ignore")
        df = df.dropna()
        if df[self.target_column].dtype == 'object':
            df[self.target_column] = pd.factorize(df[self.target_column])[0]
        X = df.drop(columns=[self.target_column]).values
        y = df[[self.target_column]].values
        # Normalize features.
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
