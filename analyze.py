import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
import re
import os,sys
import pprint

pd.options.mode.chained_assignment = None #Disable bad warning
pp = pprint.PrettyPrinter(depth=6)

bws = [
    #"io500__score",
    "ior__easy_write",
    "ior__easy_read",
    "ior__hard_write",
    "ior__hard_read",
    "mdtest__easy_write",
    "mdtest__easy_stat",
    "mdtest__easy_delete",
    "mdtest__hard_write",
    "mdtest__hard_read",
    "mdtest__hard_stat",
    "mdtest__hard_delete",
    "find__easy",
]

norm_scores = [
    #"io500__score_norm",
    "ior__easy_write_norm",
    "ior__easy_read_norm",
    "ior__hard_write_norm",
    "ior__hard_read_norm",
    "mdtest__easy_write_norm",
    "mdtest__easy_stat_norm",
    "mdtest__easy_delete_norm",
    "mdtest__hard_write_norm",
    "mdtest__hard_read_norm",
    "mdtest__hard_stat_norm",
    "mdtest__hard_delete_norm",
    "find__easy_norm",
]

numerical_factors = [
    "information__client_nodes",
    "information__md_nodes",
    "information__ds_nodes",

    #"information__client_procs_per_node",
    #"information__client_total_procs",

    #"information__md_volatile_memory_capacity",
    #"information__md_storage_devices",

    #"information__ds_volatile_memory_capacity",
    #"information__ds_storage_devices",
]

categorical_factors = [
    #"information__filesystem_type",
    #"information__client_operating_system",
    #"information__md_network",
    "information__md_storage_type",
    "information__md_storage_interface",

    #"information__ds_network",
    "information__ds_storage_type",
    "information__ds_storage_interface",
]

factors = categorical_factors + numerical_factors

def drop_duplicates(df):
    factors = [
        "information__list_id",
        "information__system",
        "information__institution",
        "information__storage_vendor",
        "information__submission_date",
        "information__storage_install_date",
        "information__filesystem_type",
        "information__filesystem_version",
        "information__client_nodes",
        "information__client_total_procs",
        "information__client_procs_per_node",
        "information__client_operating_system",
        "information__client_operating_system_version",
        "information__client_kernel_version",
        "information__md_nodes",
        "information__md_storage_devices",
        "information__md_volatile_memory_capacity",
        "information__md_storage_type",
        "information__md_storage_interface",
        "information__md_network",
        "information__ds_nodes",
        "information__ds_storage_devices",
        "information__ds_volatile_memory_capacity",
        "information__ds_storage_type",
        "information__ds_storage_interface",
        "information__ds_network",
    ]
    return df.groupby(factors).max().reset_index()

def size_to_gb(df, cols):
    for index, row in df.iterrows():
        for col in cols:
            if row[col] == -1:
                continue
            r = re.search("([0-9]+)([GT])[bB]", row[col])
            if r.group(2) == "G":
                row[col] = int(r.group(1))
            if r.group(2) == "T":
                row[col] = int(r.group(1))*1024
        df.iloc[index,:] = row
    return df

def clean_io500(df):
    #df[norm_scores] = df[bws].div((df["information__client_nodes"]/df["information__client_nodes"].max())*(df["information_net_ds_storage_devices"]/df["information_net_ds_storage_devices"].max()), axis=0)
    df = df.fillna(-1)
    #df = drop_duplicates(df)
    #df = size_to_gb(df, ["information__ds_volatile_memory_capacity", "information__md_volatile_memory_capacity"])
    return df

def grp_key(factors, vals):
    return {factor : val for factor,val in zip(factors, vals)}

def basic_stats(grps):
    mean_df = grps.mean().reset_index()
    mean_df = mean_df.rename(columns={col: "mean_" + col for col in mean_df.columns})
    std_df = grps.std().reset_index()
    std_df = std_df.rename(columns={col: "std_" + col for col in std_df.columns})
    return pd.concat([mean_df, std_df], axis=1)

def cloud_v_hpc():
    df = clean_io500(pd.read_csv("views/cloud_v_hpc.csv"))#[["information__is_cloud"] + norm_scores]
    grps = df.groupby(["information__is_cloud"])
    stats = basic_stats(grps)
    stats.to_csv("stats/cloud_v_hpc_stats.csv")

def storage_type_comparison():
    df = clean_io500(pd.read_csv("views/storage_type_perf.csv"))
    grps = df.groupby(["information__ds_storage_type", "information__ds_storage_interface"])
    stats = basic_stats(grps)
    stats.to_csv("stats/storage_type_stats.csv")

def storage_type_cost_comparison():
    df = clean_io500(pd.read_csv("views/storage_type_cost.csv"))
    grps = df.groupby(["information__ds_storage_type", "information__ds_storage_interface"])
    stats = basic_stats(grps)
    stats.to_csv("stats/storage_type_cost_stats.csv")

storage_type_cost_comparison()
