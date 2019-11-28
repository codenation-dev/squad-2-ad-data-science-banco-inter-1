import numpy as np
import pandas as pd
from config import Configure
class Preprocessing:
    def __init__(self):
        # loading dataframes
        self.df1 = pd.read_csv('../data/estaticos_portfolio1.csv', index_col=0)
        self.df2 = pd.read_csv('../data/estaticos_portfolio2.csv', index_col=0)
        self.df3 = pd.read_csv('../data/estaticos_portfolio3.csv', index_col=0)
        self.mkt = pd.read_csv('../data/estaticos_market.csv', index_col=0)
        # merging dataframes
        self.df2 = pd.merge(self.df2, self.mkt, on='id', how='inner')
        self.df3 = pd.merge(self.df3, self.mkt, on='id', how='inner')
        # loading configurations
        self.configurations = Configure()
        self.configurations.set_pre_processing_params()

    def drop_eda(self):
        self.mkt.drop(self.configurations.pre_processing_params['EDA'], axis=1)

    def filter_nan(self):
        threshold = self.configurations.pre_processing_params['threshold']
        # null reg per column
        nan_num = (self.mkt.isnull().sum(axis=0)) / self.mkt.shape[0]
        # get columns whose have more nulls than defined threshold
        column_nans = self.mkt.columns[nan_num >= threshold]
        # drop values
        self.mkt.drop(column_nans, axis=1)


