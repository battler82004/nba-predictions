# NBA Data Fetcher
# James Taddei
# 2024-10-23

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

    # This function will fetch box score statistics for a given game, specified by the game_id
    # We get this game_id from the game logs
    # data_type can be "advanced", "traditional", "misc", "hustle", "track"
    # These data types represent different endpoints we call to collect data
    # We call 5 different ones because each provides different statistics about the team's performance in that game
    def fetch_box_score(self, game_id, data_type):
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

                # Call the correct API with the parameters we specified
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
        # We will fill this in later to handle different kinds of exceptions that may arise when calling the APIs
        except:
            pass

        return


if __name__ == "__main__":
    # Specify the seasons we want to collect data for
    seasons = ["2023-24", "2022-23", "2021-22"]

    # Create new NBADataFetcher for each season
    for season in seasons:
        fetcher = NBADataFetcher(season)
        fetcher.fetch_and_save_all_data()  # This function will be completed later
