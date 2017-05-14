#!/usr/local/bin/python

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
from sklearn import decomposition

class APP():
    def __init__(self, train_data, test_data, user_data,
                 Ad_data, position_data, app_category_data,
                 user_app_action_data, user_installedapps_data):
        self._train_dataFrame = pd.read_csv(train_data)
        self._test_dataFrame = pd.read_csv(test_data)
        self._user_dataFrame = pd.read_csv(user_data)
        self._Ad_dataFrame = pd.read_csv(Ad_data)
        self._position_dataFrame = pd.read_csv(position_data)
        self._app_category_dataFrame = pd.read_csv(app_category_data)
        self._user_app_action_dataFrame = pd.read_csv(user_app_action_data)

    def _normizeData(self, dataFrame):
        means = self._df.loc[:, Numeric_columns].mean()
        stdev = self._df.loc[:, Numeric_columns].std()
        self._df.loc[:, Numeric_columns] = (self._df.loc[:, Numeric_columns] - means) / stdev
