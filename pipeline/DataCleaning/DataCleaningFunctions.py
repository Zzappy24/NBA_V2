import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from path.path import CURATED_DATA_DIR, CURATED_DATA_DIR_TEMP, RAW_DATA_DIR_TEMP

def read_data():
    filename = list(RAW_DATA_DIR_TEMP.glob('*.csv'))[0]
    try :
        players = pd.read_csv(filename,sep =";", encoding='Windows-1252')
        if len(players.columns) == 1:
            players = pd.read_csv(filename,sep =",", encoding='Windows-1252')
    except Exception:
        players = pd.read_csv(filename,sep =";", encoding='utf-8')
        if len(players.columns) == 1:
            players = pd.read_csv(filename,sep =",", encoding='utf-8')
    assert_not_null(players)
    return players

def assert_not_null(df):
    assert sum(df.isnull().sum()) < 1000, "There are not null values in the dataset"

def transform_data(players):
    players["EFF"] = players.PTS + players.TRB + players.AST + players.STL + players.BLK - (players.FGA - players.FG) - (players.FTA - players.FT) - players.TOV
    players['TS%'] = np.where((2 * (players['FGA'] + 0.44 * players['FTA'])) != 0, players['PTS'] / (2 * (players['FGA'] + 0.44 * players['FTA'])), 0)
    players["position"] = players.Pos.map({"PG": "Backcourt", "SG": "Backcourt", "SF": "Wing", "SF-PF": "Wing", "PF": "Big", "C": "Big", })
    players = players.fillna(0)
    return players


def write_csv_cleaned(df, timestamp):
    df.to_csv(f"{CURATED_DATA_DIR}/curated_data_{timestamp}.csv", index=False)


def write_csv_cleaned_temp(df, timestamp):
    df.to_csv(f"{CURATED_DATA_DIR_TEMP}/curated_data_temp_{timestamp}.csv", index=False)  


def main_cleaning():
    df = read_data()
    df = transform_data(df)
    #write_csv_cleaned_temp(df)
    return df