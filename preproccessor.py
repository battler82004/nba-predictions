# Preprocessor
# James Taddei

import pandas as pd
import numpy as np
from datetime import date

class Preproccessor:
    def __init__(self, seasons, span, shift):
        self.seasons = seasons
        self.games = pd.DataFrame()
        print("Loading games")
        self.load_all_games()
        self.team_stats = pd.DataFrame()
        self.span = span
        self.shift = shift

        print("Loading team data")
        self.load_team_data()

    def load_all_games(self):
        seasons = []
        for season in self.seasons:
            df_games = pd.read_csv(f"{season}_all_games.csv")
            seasons.append(df_games)
        self.games = pd.concat(seasons)
        self.games["GAME_DATE"] = self.games["GAME_DATE"].apply(
            lambda x: date(*map(int, x.split("-")))
        ) # puts dates in datetime format if they are not already

    def load_team_data(self):
        seasons = []
        for season in self.seasons:
            df_games = pd.read_csv(f"{season}_all_games.csv")
            df_advanced = pd.read_csv(f"{season}_advanced_stats.csv")[
                [
                    "gameId",
                    "teamTricode",
                    "estimatedOffensiveRating",
                    "offensiveRating",
                    "estimatedDefensiveRating",
                    "defensiveRating",
                    "estimatedNetRating",
                    "netRating",
                    "assistPercentage",
                    "assistToTurnover",
                    "assistRatio",
                    "offensiveReboundPercentage",
                    "defensiveReboundPercentage",
                    "reboundPercentage",
                    "turnoverRatio",
                    "effectiveFieldGoalPercentage",
                    "trueShootingPercentage",
                    "usagePercentage",
                    "estimatedUsagePercentage",
                    "estimatedPace",
                    "pace",
                    "pacePer40",
                    "possessions",
                    "PIE",
                ]
            ]
            df_traditional = pd.read_csv(f"{season}_traditional_stats.csv")[
                [
                    "gameId",
                    "teamTricode",
                    "fieldGoalsMade",
                    "fieldGoalsAttempted",
                    "fieldGoalsPercentage",
                    "threePointersMade",
                    "threePointersAttempted",
                    "threePointersPercentage",
                    "freeThrowsMade",
                    "freeThrowsAttempted",
                    "freeThrowsPercentage",
                    "reboundsOffensive",
                    "reboundsDefensive",
                    "reboundsTotal",
                    "assists",
                    "steals",
                    "blocks",
                    "turnovers",
                    "foulsPersonal",
                    "points",
                    "plusMinusPoints",
                ]
            ]
            df_hustle = pd.read_csv(f"{season}_hustle_stats.csv")[
                [
                    "gameId",
                    "teamTricode",
                    "contestedShots",
                    "contestedShots2pt",
                    "contestedShots3pt",
                    "deflections",
                    "chargesDrawn",
                    "screenAssists",
                    "screenAssistPoints",
                    "looseBallsRecoveredOffensive",
                    "looseBallsRecoveredDefensive",
                    "looseBallsRecoveredTotal",
                    "offensiveBoxOuts",
                    "defensiveBoxOuts",
                    "boxOutPlayerTeamRebounds",
                    "boxOutPlayerRebounds",
                    "boxOuts",
                ]
            ]
            df_misc = pd.read_csv(f"{season}_misc_stats.csv")[
                [
                    "gameId",
                    "teamTricode",
                    "pointsOffTurnovers",
                    "pointsSecondChance",
                    "pointsFastBreak",
                    "pointsPaint",
                    "oppPointsOffTurnovers",
                    "oppPointsSecondChance",
                    "oppPointsFastBreak",
                    "oppPointsPaint",
                    "blocksAgainst",
                    "foulsDrawn",
                ]
            ]
            df_tracking = pd.read_csv(f"{season}_track_stats.csv")[
                [
                    "gameId",
                    "teamTricode",
                    "distance",
                    "reboundChancesOffensive",
                    "reboundChancesDefensive",
                    "reboundChancesTotal",
                    "touches",
                    "secondaryAssists",
                    "freeThrowAssists",
                    "passes",
                    "contestedFieldGoalsMade",
                    "contestedFieldGoalsAttempted",
                    "contestedFieldGoalPercentage",
                    "uncontestedFieldGoalsMade",
                    "uncontestedFieldGoalsAttempted",
                    "uncontestedFieldGoalsPercentage",
                    "defendedAtRimFieldGoalsMade",
                    "defendedAtRimFieldGoalsAttempted",
                    "defendedAtRimFieldGoalPercentage",
                ]
            ]

            merge_columns = ["gameId", "teamTricode"] # PK is a team's performance in a specific game
            merged_df = pd.merge(df_advanced, df_traditional, on=merge_columns, how="inner")
            merged_df = pd.merge(merged_df, df_hustle, on=merge_columns, how="inner")
            merged_df = pd.merge(merged_df, df_misc, on=merge_columns, how="inner")
            merged_df = pd.merge(merged_df, df_tracking, on=merge_columns, how="inner")
            merged_df = merged_df.drop_duplicates()

            merged_df = pd.merge(
                merged_df, df_games[["gameId", "GAME_DATE"]], on="gameId", how="inner"
            ) # grab the data of the game because it's not in any merged dataframe\

            merged_df["date"] = merged_df["GAME_DATE"].apply(
                lambda x: date(*map(int, x.split("-")))
            )
            merged_df = merged_df.drop(columns = ["GAME_DATE"])

            # Adds a new columns for the winner of the game. Works like a functional if (if condition, include first; otherwise, second) 
            df_games["winner"] = np.where(
                df_games["HOME_TEAM_PTS"] > df_games["AWAY_TEAM_PTS"],
                df_games["HOME_TEAM_ABBREVIATION"],
                df_games["AWAY_TEAM_ABBREVIATION"]
            )

            merged_df = pd.merge(
                merged_df,
                df_games[[
                    "gameId",
                    "winner",
                    "HOME_TEAM_ABBREVIATION",
                    "AWAY_TEAM_ABBREVIATION"
                ]],
                on="gameId",
                how="inner"
            )

            processed_df = self.preprocess_team_data(merged_df)
            seasons.append(processed_df)
        
        self.team_stats = pd.concat(seasons)

    def preprocess_team_data(self, df):
        grouped = df.groupby("teamTricode")
        modified_groups = []
        for _, group in grouped:
            group["game_count"] = (
                group["gameId"].expanding().count().shift(self.shift).fillna(0)
            )
            group["time_between_games"] = group["date"].diff().dt.days
            group["playoff"] = (group["game_count"] > 82).astype(int)
            group = self.generate_team_running_averages(group)

            modified_groups.append(group)
        
        return pd.concat(modified_groups)
    
    def generate_team_running_averages(self, group):
        percentage_columns = [
            # FILL IN
        ]

        # Average every column that is not non-averagable or a percentage
        averaging_columns = [
            col for col in group.columns if (col not in [
                # FILL IN
            ]
            and col not in percentage_columns)
        ]

        for col in averaging_columns:
            running_avg_col_name = f"running_avg_{col}_last_{self.span}"
            group[running_avg_col_name] = (
                group[col].ewm(span=self.span).mean().shift(self.shift) # ewm: exponential weighted mean (recently weighted higher)
            )
        
        group= group.drop(columns=averaging_columns)
        return group

if __name__ == "__main__":
    seasons = ["2024-25"]
    p50 = Preproccessor(seasons, 50, 1)
    p25 = Preproccessor(seasons, 25, 1)
    p10 = Preproccessor(seasons, 10, 1)

    common_cols = list(set(p50.team_stats.columns).intersection(set(p25.team_stats.columns)))
    p50.team_stats = pd.merge(p50.team_stats, p25.team_stats, on=common_cols, how="inner")
    p50.team_stats = pd.merge(p50.team_stats, p10.team_stats, on=common_cols, how="inner")

    print("Processing complete, saving data")
    p50.games.to_csv("all_games.csv")
    p50.teams_stats.to_csv("all_team_averages.csv")