import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline

from config import Configure
class Preprocessing:
    def __init__(self, cat_vars, num_vars, manual_vars, date_vars):
        self.cat_vars = cat_vars
        self.num_vars = num_vars
        self.manual_vars = manual_vars
        self.date_vars = date_vars
        self.preprocess = []

    def pipe_lines_creating(self):
        self.preprocess = make_column_transformer(
            (self.cat_vars, make_pipeline(SimpleImputer(strategy='most_frequent'), LabelEncoder())),
            (self.num_vars, make_pipeline(SimpleImputer(strategy='median'), StandardScaler())))




