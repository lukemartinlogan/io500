
import pandas as pd
import re

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
    features = list(pd.read_csv("features/all-features.csv").iloc[:,0])
    categorical = list(pd.read_csv("features/categorical-features.csv").iloc[:,0])
    df.loc[:,categorical] = df[categorical].fillna("None")
    df = df.fillna(-1)
    df = df.loc[df.groupby(features)["io500__score"].idxmax(),:].reset_index()
    df = size_to_gb(df, ["information__ds_volatile_memory_capacity", "information__md_volatile_memory_capacity"])
    #df = pd.get_dummies(df)
    return df

df = pd.read_csv("datasets/data-2019-2020-cleaned.csv")
df = clean_io500(df)
df.to_csv("datasets/preprocessed-2019-2020.csv", index=False)
