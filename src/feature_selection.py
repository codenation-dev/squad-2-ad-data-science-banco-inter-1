## feature_selection

import numpy as np
import pandas as pd
from config import Configure
from preprocessing import Preprocessing


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
        concat_df = [self.df3, self.df2, self.df3]
        concat_df = pd.concat(concat_df, ignore_index=True)
        concat_df.drop(self.columns_chosen+['Unnamed: 0_x', 'Unnamed: 0_y', 'id'],
                       axis=1,
                       inplace=True)
        return concat_df


def main():
    settings = Configure()
    settings.set_fs_params()
    df1 = pd.read_csv(settings.pf1_folder)
    df2 = pd.read_csv(settings.pf2_folder)
    df3 = pd.read_csv(settings.pf3_folder)
    mkt = pd.read_csv(settings.mkt_folder)
    test = FeatureSelection(df1=df1, df2=df2, df3=df3, mkt=mkt)
    test.eda_nan_columns(settings.feature_selection_params)
    columns_type = test.drop_columns().dtypes
    cat_features = columns_type[columns_type == 'object']
    num_features = columns_type[columns_type != 'object']
    pre_process = Preprocessing(cat_vars=cat_features,
                                num_vars=num_features,
                                date_vars=[],
                                manual_vars=[])
    pre_process.pipe_lines_creating()
    print(pre_process.preprocess.fit_transform(test.drop_columns()))




if __name__ == '__main__':
    main()
