# NBA Data Fetcher
# James Taddei

import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamelog,
    boxscoreadvancedv3,
    boxscoretraditionalv3,
    boxscorehustlev2,
    boxscoremiscv3,
    boxscoreplayertrackv3,
    PlayerGameLogs,
    commonteamroster,
)
from nba_api.stats.static import teams
from datetime import date
import time
from requests.exceptions import ReadTimeout
from json.decoder import JSONDecodeError


# Use this class to fetch data for a given season
class NBADataFetcher:
    def __init__(self, season):
        # Specify the season we want to collect data for
        # Season must be of format "YYYY-YY" (i.e., "2023-24")
        self.season = season

        # This will be used later to map from team id to the team's abbreviation
        # I.e., team id is some number (30 for example), abbreviation will be something like "nyk" for New York Knicks
        # This will be useful when we are processing API requests later
        self.team_dict = {
            team["id"]: team["abbreviation"] for team in teams.get_teams()
        }

        # Fetch and process game logs
        game_logs = self.fetch_league_game_logs()
        processed_game_logs = self.process_game_logs(game_logs)
        self.processed_game_logs = processed_game_logs

    def fetch_league_game_logs(self):
        # Gets game information for all regular season games for specified season
        game_log = leaguegamelog.LeagueGameLog(
            season=self.season, season_type_all_star="Regular Season"
        )

        # Gets game information for all playoff games for specified season
        playoff_log = leaguegamelog.LeagueGameLog(
            season=self.season, season_type_all_star="Playoffs"
        )

        # Concatenates the regular season games and playoff games and returns a pandas DataFrame containing these
        return pd.concat(
            [game_log.get_data_frames()[0], playoff_log.get_data_frames()[0]]
        )
    
    def process_game_logs(self, game_logs):
        # Reformat dates as date object
        game_logs["GAME_DATE"] = game_logs["GAME_DATE"].apply(
            # For each date in GAME_DATE: split it by dashes, use map to make each an int, apply date function
            lambda x: date(*map(int, x.split("-")))
        )

        # Filters out logs that could cause errors
        game_logs = game_logs[game_logs["GAME_DATE"] != date.today()] # games from today may be incomplete
        game_logs = game_logs.drop_duplicates(subset=["GAME_ID","TEAM_ID"]) # removes duplicates

        # Home logs include "vs." in matchup while away games use "@"
        home_logs = game_logs[game_logs["MATCHUP"].str.contains(" vs. ").copy()]
        away_logs = game_logs[game_logs["MATCHUP"].str.contains(" @ ").copy()]

        # Prepare home and away logs for merging
        home_logs = home_logs.rename(columns={
            "TEAM_ID": "HOME_TEAM_ID",
            "TEAM_ABBREVIATION": "HOME_TEAM_ABBREVIATION",
            "PTS": "HOME_TEAM_PTS"}
        )
        away_logs = away_logs.rename(columns={
            "TEAM_ID": "AWAY_TEAM_ID",
            "TEAM_ABBREVIATION": "AWAY_TEAM_ABBREVIATION",
            "PTS": "AWAY_TEAM_PTS"}
        )

        # Merge home and away logs
        merged_df = pd.merge(home_logs,away_logs,on=["GAME_ID", "GAME_DATE"], suffixes=("_home", "_away"))
        merged_df = merged_df[[
            "GAME_ID",
            "GAME_DATE",
            "HOME_TEAM_ABBREVIATION",
            "HOME_TEAM_PTS",
            "AWAY_TEAM_ABBREVIATION",
            "AWAY_TEAM_PTS"]]
        merged_df = merged_df.rename(columns={"GAME_ID": "gameId"})

        return merged_df

    # This function will fetch box score statistics for a given game, specified by the game_id
    # We get this game_id from the game logs
    # data_type can be "advanced", "traditional", "misc", "hustle", "track"
    # These data types represent different endpoints we call to collect data
    # We call 5 different ones because each provides different statistics about the team's performance in that game
    def fetch_box_score(self, game_id, data_type):
        max_retries = 10
        wait_seconds = 0.1
        for attempt in range(max_retries):
            # Use try block in case of API exception
            try:
                # For advanced, traditional, and miscellaneous box score API calls, we need to specify extra parameters
                if data_type in ["advanced", "traditional", "misc"]:
                    params = {
                        "end_period": 0,
                        "end_range": 0,
                        "game_id": game_id,
                        "range_type": 0,
                        "start_period": 0,
                        "start_range": 0,
                    }

                    # Call the correct API with the parameters we specified. ** operator unpacks params to be an arg list
                    if data_type == "advanced":
                        box_score = boxscoreadvancedv3.BoxScoreAdvancedV3(**params)
                    elif data_type == "traditional":
                        box_score = boxscoretraditionalv3.BoxScoreTraditionalV3(**params)
                    else:
                        box_score = boxscoremiscv3.BoxScoreMiscV3(**params)
                # For hustle and track box score API calls, we only need to specify the game_id we want
                else:
                    if data_type == "hustle":
                        box_score = boxscorehustlev2.BoxScoreHustleV2(game_id)
                    else:
                        box_score = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id)

                # Isolate the team data from the whole box score
                team_stats = box_score.team_stats.get_data_frame()
                print(team_stats)
                return team_stats

            # We will fill this in later to handle different kinds of exceptions that may arise when calling the APIs
            except JSONDecodeError as e:
                if attempt < max_retries:
                    print(f"JSONDecode Error Occurred. Retying for the {attempt+1} / {max_retries} attempt")
                    time.sleep(wait_seconds)
                else:
                    print("Fetching this game failed. Skipping")

if __name__ == "__main__":
    # Specify the seasons we want to collect data for
    seasons = ["2023-24", "2022-23", "2021-22"]

    # Create new NBADataFetcher for each season
    for season in seasons:
        fetcher = NBADataFetcher(season)
        print(fetcher.fetch_box_score("0022300062", "traditional"))
        fetcher.processed_game_logs.to_csv("processed_game_logs.csv")