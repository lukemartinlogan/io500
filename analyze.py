import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree


from src.dataset import Dataset
from src.curve_wrapper import CurveWrapper
from src.forest_wrapper import ForestWrapper
from src.ensemble_model import EnsembleModelRegressor
from src.dataset import Dataset

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
import re
import os,sys
import pprint

pd.options.mode.chained_assignment = None #Disable bad warning
pp = pprint.PrettyPrinter(depth=6)

def categorical_feature(df_wrap, vars):
    #Group by feature
    grp = df_wrap.groupby([
        "information__filesystem_type"
    ]).divide()

    #How tight is performance?
    pp.pprint(grp.analyze(vars))

FEATURES = Dataset().load_features("features/features.csv")
PERFORMANCE = Dataset().load_features("features/performance.csv")
df_wrap = Dataset().read_csv("datasets/preprocessed-2019-2020.csv")

#Select 10-node challenge
df_wrap.df = df_wrap.df[df_wrap.df["information__client_nodes"] == 10]
categorical_feature(df_wrap, PERFORMANCE)
