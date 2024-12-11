# Model
# James Taddei

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScalar
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

def load_data():
    df_games = pd.read_csv("all_games.csv", index_col=0)
    df_games["GAME_DATE"] = pd.to_datetime(df_games["GAME_DATE"])
    df_games["spread"] = (df_games["HOME_TEAM_PTS"] - df_games["AWAY_TEAM_PTS"]).astype(int)

    df_averages = pd.read_csv("all_team_averages.csv", index_col=0)
    df_averages = df_averages.drop(columns=["date"])
    df_averages = df_averages[df_averages["game_count"] >= 10] # remove first 9 games from stats (more outliers)

    return df_games, df_averages

def generate_home_away_data(df_games, df_averages):
    # Filter out home games by finding cases where the game-team key of df_averages matches the game-home team key of df_games
    df_home = pd.merge(df_averages, df_games[["gameId", "HOME_TEAM_ABBREVIATION", "spread"]], left_on=["gameId", "teamTricode"], right_on=["gameId", "HOME_TEAM_ABBREVIATION"], how="inner")
    df_away = pd.merge(df_averages, df_games[["gameId", "AWAY_TEAM_ABBREVIATION", "spread"]], left_on=["gameId", "teamTricode"], right_on=["gameId", "AWAY_TEAM_ABBREVIATION"], how="inner")

    df_home = df_home.sort_values(by="gameId")
    df_away = df_away.sort_values(by="gameId")
    df_games = df_games.sort_values(by="gameId")

    df_home = df_home.drop(columns=["HOME_TEAM_ABBREVIATION"])
    df_away = df_away.drop(columns=["AWAY_TEAM_ABBREVIATION", "spread", "playoff"])

    df_home = pd.get_dummies(df_home, columns=["teamTricode"])
    df_away = pd.get_dummies(df_away, columns=["teamTricode"])

    spread = df_home[["gameId", "spread", "playoff"]]
    df_home = df_home.drop(columns=["spread", "playoff"])

    return df_home, df_away, spread

def generate_combined_data(df_home, df_away):
    df_combined = pd.merge(df_home, df_away, on=["gameId"], how="inner", suffixes=["_home", "_away"])
    return df_combined

def generate_diff_features(df_home, df_away):
    df_home = df_home.set_index("gameId")
    df_away = df_away.set_index("gameId")

    # Subtract columns that aren't the team tricode
    diff_columns = [col for col in df_home.columns if "teamTricode" not in col]
    diff_features = df_home[diff_columns].sub(df_away[diff_columns], fill_value=0)

    diff_features = diff_features.reset_index()
    diff_features.columns = ["gameId"] + [f"diff_" for col in diff_features if col != "gamId"]
    return diff_features

def generate_full_vector(df_combined, diff_features, spread):
    df_combined = df_combined.merge(diff_features, on="gameId", how="left")
    df_combined = pd.merge(df_combined, spread, on="gameId", how="inner")
    df_combined = df_combined.sort_values(by="gameId")
    return df_combined

def generate_y(df_combined):
    y = df_combined[["spread"]]
    y = y.values.ravel() # treats y as a column (series) rather than df
    return y

def split_scale_data(df_games, cutoff_date, df_combined, y):
    train_mask = df_games[["GAME_DATE"] < pd.to_datetime(cutoff_date)]
    test_mask = df_games[["GAME_DATE"] >= pd.to_datetime(cutoff_date)]

    train_gameIds = df_games.loc[train_mask, "gameId"].unique()
    test_gameIds = df_games.loc[test_mask, "gameId"].unique()

    df_train = df_combined[df_combined["gameId"].isin(train_gameIds)]
    df_test = df_combined[df_combined["gameIds"].isin(test_gameIds)]
    y_train = y[df_combined["gameIds"].isin(train_gameIds)]
    y_test = y[df_combined["gameIds"].isin(test_gameIds)]

    # FILL IN

def generate_data(cutoff_date):
    # cutoff_date helps split training and test data. Everything before is training, everything after is test
    df_games, df_average = load_data()
    df_home, df_away, spread = generate_home_away_data(df_games, df_average)
    df_combined = generate_combined_data(df_home, df_away)
    diff_features = generate_diff_features(df_home, df_away)
    df_combined = generate_full_vector(df_combined, diff_features, spread)
    y = generate_y(df_combined)
    df_combined = df_combined.drop(columns=["spread"])
    df_train, df_test, y_train, y_test = split_scale_data(df_games, cutoff_date, df_combined, y)

    return df_train, df_test, y_train, y_test

df_train, df_test, y_train, y_test = generate_data("2024-08-01")

def train_models(n):
    num_samples = df_test.shape[0]

    models = []
    for i in range(1, n+1):
        print(f"Training model {i} / {n}")
        xgb_model = XGBRegressor(
            max_depth=random.randint(4,10),
            learning_rate=random.randrange(5, 20, 1)/1000.0,
            n_estimators=random.randint(700, 1500),
            eval_metric="rmse",
            objective="reg:squarederror",
            alpha=random.randrange(5, 20, 1)/1000.0,
            early_stopping_rounds=50,
            device="cuda"
        )
        xgb_model.fit(df_train, y_train, eval_set=[(df_test, y_test)], verbose=True)

        models.append(xgb_model)
        pickle.diump(xgb_model. open(f"spread_model_{i}.pkl", "wb"))

train_models(5)