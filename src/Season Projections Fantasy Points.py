"""
This will create season long fantasy point predictions and test the prediction against actuals
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

combined_df = pd.read_csv('/Users/scotthoran/Library/CloudStorage/OneDrive-Personal/Documents/Daily Fantasy - Active/DFS Python Build/Transformation/combined_player_transformed_data.csv')
columns_to_clean = ['passing_yds', 'passing_td',
       'passing_att', 'passing_cmp', 'receiving_yds', 'receiving_td',
       'receiving_rec', 'receiving_tar', 'rushing_att', 'rushing_td',
       'rushing_yds', 'FantasyPoints']
combined_df[columns_to_clean] = combined_df[columns_to_clean].fillna(0)


season_columns = ["player_id", "name", "position_id", "season", "FantasyPoints"]
regular_season_df = combined_df[season_columns]

season_agg_df = regular_season_df.groupby(
    ["player_id", "name", "position_id", "season"]).sum()
season_agg_df = season_agg_df.reset_index()

#creating blank 2023 rows for prediction
season_agg_df_2023 = season_agg_df.copy()
players_with_fantasy_points_2022 = season_agg_df_2023[season_agg_df_2023['season'] == 2022]['player_id'].unique()
season_agg_df_2023 = season_agg_df_2023[season_agg_df_2023['player_id'].isin(players_with_fantasy_points_2022)].drop(columns=['FantasyPoints'])
season_agg_df_2023['season'] = 2023 #this will be blank in 2023
season_agg_df = pd.concat([season_agg_df,season_agg_df_2023], ignore_index = True)
season_agg_df.sort_values(['player_id','season']).reset_index(drop=True)


#creating auto-regressive model to predict full year fantasy production based off historical data
#need to add in "lag" features to put past season avg's and totals on the same line
#https://gridironai.com/football/blog/article/Modeling_With_NFL_Data_A_Quick_Start_Guide/75
lag_features = ['FantasyPoints']
for lag in range (1,6):
    shifted = season_agg_df.groupby('player_id').shift(lag)
    season_agg_df = season_agg_df.assign(**{f"lag_{column}_{lag}": shifted[column] 
                          for column in lag_features})
season_agg_df = season_agg_df.fillna(-1)

#to build model, need test and training data
#Will use 2021 and 2022 seasons
train_df = season_agg_df[season_agg_df["season"] < 2021].reset_index(drop=True)
test_df = season_agg_df[season_agg_df["season"] >= 2022].reset_index(drop=True)

#used to check correlation of the lag factors (more about data exploration)
#sns.heatmap(train_df.corr()[["FantasyPoints"]], annot=True) - use to display correlation

#defining modeling pipeline now

features = [
    "lag_FantasyPoints_1", 
    "lag_FantasyPoints_2", 
]

#this creates rules for the model's pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", ElasticNet())
])

param_grid = {
    "lr__alpha": [0.001, 0.01, 0.1, 1, 10], # regularization strength
    "lr__l1_ratio": [0.1, 0.2, 0.5, 0.8, 0.9, 1.0] # type of regulariztion
}
grid = GridSearchCV(pipeline, 
                    n_jobs=-1,
                    param_grid=param_grid, 
                    scoring="neg_mean_absolute_error")
grid.fit(train_df[features], train_df["FantasyPoints"])
print(f"Found best parameters: {grid.best_params_}") 

train_mae = mean_absolute_error(train_df["FantasyPoints"]
                                ,grid.predict(train_df[features]))

test_df["predicted_FantasyPoints"] = grid.predict(test_df[features])
test_mae = mean_absolute_error(test_df["FantasyPoints"]
                                ,test_df["predicted_FantasyPoints"])

print(f"train mae: {train_mae}, test mae: {test_mae}")

#train mae: 45.23459303446201, test mae: 44.973591692031775 - for 2022 this was the output meaning we are ~45 points off on prediction

draft_df = test_df[test_df["season"] == 2023].sort_values("predicted_FantasyPoints"
                                                            , ascending=False)
positions, position_dfs = zip(*draft_df.groupby("position_id"))

rows = [{} for _ in range(10)]
for position, position_df in draft_df.groupby("position_id"):
    for i, (_, player) in enumerate(position_df.head(10).iterrows()):
        rows[i][position] = player["name"]
        rows[i][f"{position}_predicted"] = player["predicted_FantasyPoints"]
        rows[i][f"{position}_points"] = player["FantasyPoints"]

pd.DataFrame(rows)

