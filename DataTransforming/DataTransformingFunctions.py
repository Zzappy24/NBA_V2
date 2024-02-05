import pandas as pd
import numpy as np
from pathlib import Path
from path.path import CURATED_DATA_DIR_TEMP, TRANFORMED_DATA_DIR, TRANFORMED_DATA_DIR_TEMP
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


def read_data():
    filename = list(CURATED_DATA_DIR_TEMP.glob('*.csv'))[0]
    print(f"Runnung on file: {filename}")
    players = pd.read_csv(filename,sep =";", encoding='Windows-1252')
    if len(players.columns) == 1:
        players = pd.read_csv(filename,sep =",", encoding='Windows-1252')
    assert_not_null(players)
    return players

def assert_not_null(df):
    assert sum(df.isnull().sum()) < 30, f"There are not null values in the dataset {sum(df.isnull().sum())}"


def transform_data(players):
    ages = players.Age.describe().round(decimals=1) # used to specify the first 25%, defining what is a young player
    young_age = ages["25%"]
    futur_super_star_def = f"(EFF >= 12) & (PTS >= 15) & (Age <= {young_age})"
    df_star = (players.query(futur_super_star_def).sort_values("EFF", ascending=False).sort_values(["Age", "EFF"], ascending=True))
    players['future_star'] = players['Player'].isin(df_star['Player']).astype(int)
    return players



def date_timestamp():
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    return int(ts)

def write_csv_tranformed(df):
    df.to_csv(f"{TRANFORMED_DATA_DIR}/transformed_data_{date_timestamp()}.csv", index=False)

def write_csv_transformed_temp(df):
    df.to_csv(f"{TRANFORMED_DATA_DIR_TEMP}/transformed_data_temp_{date_timestamp()}.csv", index=False)



def main_transforming():
    df = read_data()
    df = transform_data(df)
    #write_csv_transformed_temp(df)
    return df

