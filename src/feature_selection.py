## feature_selection

import numpy as np
import pandas as pd
from config import Configure
from preprocessing import Preprocessing

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class FeatureSelection:
    def __init__(self, mkt,  df1, df2, df3):
        self.mkt = mkt
        self.df1 = df1
        self.df2 = pd.merge(df2, mkt, on='id', how='inner')
        self.df3 = pd.merge(df3, mkt, on='id', how='inner')
        self.columns_chosen = []

    def eda_nan_columns(self, params):
        # null reg per column
        nan_num = (self.mkt.isnull().sum(axis=0)) / self.mkt.shape[0]
        # get columns whose have more nulls than defined threshold
        column_nans = self.mkt.columns[nan_num >= params['threshold']]
        self.columns_chosen = list(params['EDA']) + list(column_nans)

    def drop_columns(self):
        self.df1['target'] = np.ones(self.df1.shape[0]).astype(np.int)
        self.df2['target'] = 2*np.ones(self.df2.shape[0]).astype(np.int)
        self.df3['target'] = 3*np.ones(self.df3.shape[0]).astype(np.int)
        concat_df = [self.df1, self.df2, self.df3]
        concat_df = pd.concat(concat_df, ignore_index=True)
        concat_df.drop(self.columns_chosen+['Unnamed: 0_x', 'Unnamed: 0_y', 'id'],
                       axis=1,
                       inplace=True)
        return concat_df

    @staticmethod
    def feature_selection_method():
        print('\n Aplicando algoritmo de seleção de parâmetros')
        settings = Configure()
        settings.set_fs_params()
        df1 = pd.read_csv(settings.pf1_folder)
        df2 = pd.read_csv(settings.pf2_folder)
        df3 = pd.read_csv(settings.pf3_folder)
        mkt = pd.read_csv(settings.mkt_folder)
        data = FeatureSelection(df1=df1, df2=df2, df3=df3, mkt=mkt)
        data.eda_nan_columns(settings.feature_selection_params)
        data = data.drop_columns()
        y = data['target'].values
        X = data.drop(['target', 'Unnamed: 0'], axis=1)
        columns_type = X.dtypes
        manual_features = ['de_saude_tributaria', 'de_nivel_atividade']
        cat_features = columns_type[(columns_type == 'object')].keys()
        cat_features = cat_features.drop(manual_features)
        bool_features = columns_type[(columns_type == 'bool')].keys()
        num_features = columns_type[(columns_type != 'object') & (columns_type != 'bool')].keys()
        X[bool_features] = X[bool_features].astype('category')
        pre_process = Preprocessing(cat_vars=cat_features,
                                    num_vars=num_features,
                                    bool_vars=bool_features,
                                    manual_vars=manual_features)
        return mkt[pre_process.feature_selection_apply(X, y).values]

