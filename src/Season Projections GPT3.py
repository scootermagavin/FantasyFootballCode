"""
Steps to Dig in and Update DFS Algorithm
1. Get weekly Data in proper format
2. Get reliable NFL betting data (excel add-in?) 
3. Find Total Passes and Total Plays based on Betting Day
4. Allocate those Passes and Rushes to players
5. Find avg rates over previous games
6. Multiply rates by allocated rushes, passes, targets to get points
"""
"""
Steps to open and run code:
1. Open virtual environment in VSC (Python 3.11.3 .venv:venv)
2. Write "pip3 install -r requirements.txt" in this folder to grab the appropriate module needed to operate the code
"""
#from tkinter.tix import COLUMN
from operator import index
import pandas as pd
import matplotlib as plt
import seaborn as sns
import player_data_processing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

#For use when running the data compilation code and when there's an updated 2022 dataset
#input_folder = '/Users/scotthoran/Library/CloudStorage/OneDrive-Personal/Documents/Daily Fantasy - Active/DFS Python Build/Data/'
#output_folder = '/Users/scotthoran/Library/CloudStorage/OneDrive-Personal/Documents/Daily Fantasy - Active/DFS Python Build/Transformation/'
#use the above variables in calling the player_data_processing function by removing the following comment block
#player_data_processing.process_and_combine_data(input_folder,output_folder)

#need to do a bit of data exploring and machine learning. 
#Expected pass totals based on vegas odds

# Read the player data from CSV
combined_df = pd.read_csv('/Users/scotthoran/Library/CloudStorage/OneDrive-Personal/Documents/Daily Fantasy - Active/DFS Python Build/Transformation/combined_player_transformed_data.csv')
combined_df['yard_per_completion'] = combined_df['passing_yds'] / combined_df['passing_cmp']

columns_to_clean = ['passing_yds', 'passing_td',
       'passing_att', 'passing_cmp', 'receiving_yds', 'receiving_td',
       'receiving_rec', 'receiving_tar', 'rushing_att', 'rushing_td',
       'rushing_yds', 'yard_per_completion', 'FantasyPoints']
combined_df[columns_to_clean] = combined_df[columns_to_clean].fillna(0)

weekly_columns = ["player_id", "name", "position_id", "season", 'week', 'franchise_id', 'passing_cmp','passing_yds','yard_per_completion','receiving_rec','receiving_yds','receiving_tar','rushing_att','rushing_yds','FantasyPoints']
season_agg_df = combined_df[weekly_columns]

weekly_columns_to_clean = ['passing_cmp','passing_yds','yard_per_completion','receiving_rec','receiving_yds','receiving_tar','rushing_att','rushing_yds','FantasyPoints']

# Creating blank 2023 rows for prediction
regular_season_df_2023 = season_agg_df.copy()
players_with_fantasy_points_2022 = regular_season_df_2023[regular_season_df_2023['season'] == 2022]['player_id'].unique()

regular_season_df_2023 = regular_season_df_2023[regular_season_df_2023['player_id'].isin(players_with_fantasy_points_2022)].drop(columns=weekly_columns_to_clean)
regular_season_df_2023['season'] = 2023  # This will be blank in 2023
season_agg_df = pd.concat([season_agg_df, regular_season_df_2023], ignore_index=True)
season_agg_df.sort_values(['player_id', 'season', 'week'], inplace=True)

# Read opponent ranking data from CSV (assuming it has columns like 'season', 'week', 'franchise_id', 'opponent_franchise_id', and defensive stats)
opponent_ranking_df = pd.read_csv('/Users/scotthoran/Library/CloudStorage/OneDrive-Personal/Documents/Daily Fantasy - Active/DFS Python Build/Data/team_data.csv')

# Join regular_season_df and opponent_ranking_df using franchise_id and opponent_franchise_id
regular_season_df = pd.merge(season_agg_df, opponent_ranking_df,
                             left_on=['season', 'week', 'franchise_id'],
                             right_on=['season', 'week', 'opponent_team_pfr_franchise_code_id'],
                             suffixes=('', '_opponent'))

# Drop unnecessary columns from the merged DataFrame
regular_season_df.drop(columns=['opponent_franchise_id'], inplace=True)

# To build the model, need test and training data
# Will use 2021 and 2022 seasons
train_df = regular_season_df[regular_season_df["season"] < 2021].reset_index(drop=True)
test_df = regular_season_df[regular_season_df["season"] >= 2022].reset_index(drop=True)

#'/Users/scotthoran/Library/CloudStorage/OneDrive-Personal/Documents/Daily Fantasy - Active/DFS Python Build/Data/team_data.csv'

# Defining modeling pipeline now
features = [
    "lag_passing_cmp_1", "lag_passing_cmp_2",
    "lag_passing_yds_1", "lag_passing_yds_2",
    "lag_yard_per_completion_1", "lag_yard_per_completion_2",
    "lag_receiving_rec_1", "lag_receiving_rec_2",
    "lag_receiving_yds_1", "lag_receiving_yds_2",
    "lag_rushing_att_1", "lag_rushing_att_2",
    "lag_rushing_yds_1", "lag_rushing_yds_2",
    "lag_passing_cmp_avg", "lag_passing_yds_avg",
    "lag_yard_per_completion_avg", "lag_receiving_rec_avg",
    "lag_receiving_yds_avg", "lag_rushing_att_avg",
    "lag_rushing_yds_avg"
]

# This creates rules for the model's pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", ElasticNet())
])

param_grid = {
    "lr__alpha": [0.001, 0.01, 0.1, 1, 10],  # regularization strength
    "lr__l1_ratio": [0.1, 0.2, 0.5, 0.8, 0.9, 1.0]  # type of regularization
}
grid = GridSearchCV(pipeline,
                    n_jobs=-1,
                    param_grid=param_grid,
                    scoring="neg_mean_absolute_error")
grid.fit(train_df[features], train_df["FantasyPoints"])

train_mae = mean_absolute_error(train_df["FantasyPoints"], grid.predict(train_df[features]))
test_df["predicted_FantasyPoints"] = grid.predict(test_df[features])
test_mae = mean_absolute_error(test_df["FantasyPoints"], test_df["predicted_FantasyPoints"])

print(f"train mae: {train_mae}, test mae: {test_mae}")

# For 2023, this will be the output, meaning we are ~45 points off on the prediction
# train mae: 45.23459303446201, test mae: 44.973591692031775

draft_df = test_df[test_df["season"] == 2023].sort_values("predicted_FantasyPoints", ascending=False)
positions, position_dfs = zip(*draft_df.groupby("position_id"))

rows = [{} for _ in range(10)]
for position, position_df in draft_df.groupby("position_id"):
    for i, (_, player) in enumerate(position_df.head(10).iterrows()):
        rows[i][position] = player["name"]
        rows[i][f"{position}_predicted"] = player["predicted_FantasyPoints"]
        rows[i][f"{position}_points"] = player["FantasyPoints"]
