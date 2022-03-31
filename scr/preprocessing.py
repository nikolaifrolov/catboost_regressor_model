import pandas as pd
import numpy as np

class PrepareData:
    def __init__(self, filename, cat_features, num_features, target):
        self.dataframe = pd.read_csv(filename,usecols=[*num_features, *cat_features, target])
        self.target = target
        self.cat_features = cat_features
        self.num_features = num_features
        self.train_data = None
        self.test_data = None
        self.train_target = None


    def to_float(self, column):
        df = self.dataframe.copy()
        df[column].replace({'Not Available': np.nan, 'nan': np.nan, 'NaN': np.nan}, inplace=True)
        return df[column].astype(float).fillna(method='ffill')
    
    def to_string(self, column):
        df = self.dataframe.copy()
        return df[column].astype(str)

    def prepare_data(self):
        for column in self.cat_features:
            self.dataframe[column] = self.to_string(column)
            
        for column in self.num_features:
            self.dataframe[column] = self.to_float(column)
            
        self.train_data = self.dataframe[self.dataframe[self.target] != 'Not Available']
        self.test_data = self.dataframe[self.dataframe[self.target] == 'Not Available']
        self.test_data.drop(self.target, axis=1, inplace=True)
        self.train_target = self.train_data.pop(self.target)
        
        return self.train_data, self.test_data, self.train_target
        
    