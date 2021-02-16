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
from scipy.stats import ttest_ind


import numpy as np
import re
import os,sys
import pprint

pd.options.mode.chained_assignment = None #Disable bad warning
pp = pprint.PrettyPrinter(depth=6)

def categorical_feature(df_wrap, vars):
    #Group by feature
    grp = df_wrap.groupby([
        "information__filesystem_type",
        #"information__ds_storage_type",
        #"information__ds_storage_interface"
    ]).divide(features=[
        "information__filesystem_type",
        #"information__ds_storage_type",
        #"information__ds_storage_interface"
    ])

    #How tight is performance?
    all_analysis = grp.analyze(vars)
    for var,analysis in all_analysis.items():
        pd.DataFrame(analysis).to_csv("datasets/basic_stat_{}.csv".format(var))

    #pd.DataFrame({l1:{l2:ttest_ind(df1[vars],df2[vars])[1][0] for l2,df2 in grp.clusters.items()} for l1,df1 in grp.clusters.items()}).round(decimals=3).to_csv("datasets/t_mat.csv")

def random_forest(df_wrap, features, vars):
    reg = ForestWrapper(RandomForestRegressor(n_estimators=10, max_leaf_nodes=8, random_state=1, verbose=0))
    reg.fit(df_wrap.df[features], df_wrap.df[vars])
    print(reg.fitness_)
    print(reg.feature_importances_)

    reg = CurveWrapper(LinearRegression(fit_intercept=False))
    reg.fit(df_wrap.df[features], df_wrap.df[vars])
    print(reg.fitness_)
    print(reg.feature_importances_)
    print(reg.model.coef_)

FEATURES = Dataset().load_features("features/features.csv")
PERFORMANCE = Dataset().load_features("features/performance.csv")
df_wrap = Dataset().read_csv("datasets/preprocessed-2019-2020.csv")

print(PERFORMANCE)

#Select 10-node challenge
#df_wrap.df = df_wrap.df[df_wrap.df["information__client_nodes"] == 10]
#categorical_feature(df_wrap, PERFORMANCE)

random_forest(df_wrap, FEATURES, PERFORMANCE)
