from asyncore import read
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from settings import *


class CatBoostModel:
    def __init__(self, train_data, train_target, predict_data, params, plot):
        self.train_data = train_data
        self.train_target = train_target
        self.predict_data = predict_data
        self.params = params
        self.plot = plot
        self.model = None

    def train(self):
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(self.train_data, self.train_target,
                       cat_features=CAT_FEATURES, plot=self.plot)
        self.plot_feature_importance(
            self.model.get_feature_importance(), self.train_data.columns, 'CATBOOST')

    def predict(self):
        df = pd.read_csv(FILE_NAME)
        predict = self.model.predict(self.predict_data)
        df.loc[df['ENERGY STAR Score'] == 'Not Available', 'ENERGY STAR Score'] = predict
        return df

    def predict_proba(self):
        return self.model.predict_proba(self.predict_data)

    def plot_feature_importance(self, importance, names, model_type):

        feature_importance = np.array(importance)
        feature_names = np.array(names)

        data = {'feature_names': feature_names,
                'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)
        fi_df.sort_values(by=['feature_importance'],
                          ascending=False, inplace=True)

        plt.figure(figsize=(10, 8))
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
        plt.title(model_type + ' FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        plt.show()

    def get_model(self):
        return self.model

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = CatBoostRegressor().load_model(path)
