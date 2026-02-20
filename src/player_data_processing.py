#from tkinter.tix import COLUMN
from operator import index
import pandas as pd
import matplotlib as plt
import seaborn as sns
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

#Defining a function to be called elsewhere, these don't need to be defined yet
#input_folder = '/Users/scotthoran/Library/CloudStorage/OneDrive-Personal/Documents/Daily Fantasy - Active/DFS Python Build/Data/'
#output_folder = '/Users/scotthoran/Library/CloudStorage/OneDrive-Personal/Documents/Daily Fantasy - Active/DFS Python Build/Transformation/'


def process_and_combine_data(input_folder,output_folder):
    all_transformed_dfs = []
    
    fantasy_weights = {
    "passing_int": -2,
    "passing_yds": 0.04,
    "passing_two_pt_md": 2,
    "passing_td": 4,
    "receiving_yds": 0.1,
    "receiving_td": 6,
    "receiving_rec": 1, # set to 0 for non-PPR
    "rushing_td": 6,
    "rushing_yds": 0.1,
}

    def get_fantasy_points(game):
        return sum(game[column] * weight for column, weight in fantasy_weights.items()
               if not pd.isna(game[column])
               )

    for year in range (2000,2024):
        input_file_path = os.path.join(input_folder, f'player_data_{year}.csv')

        # Load data from the current year's CSV file
        df = pd.read_csv(input_file_path)
        
        #sticking to regular season and skill players
        df = df[(df["playoffs"] == 0) & 
        ~df["position_id"].isin({"OL",
                                "DEF",
                                "P", 
                                "K"})].reset_index(drop=True)
        
        df['FantasyPoints'] = df.apply(
        get_fantasy_points, axis="columns")
        
        df = df[[
        'player_id', 'name', 'franchise_id','position_id','season','week',
        'has_missed_game_from_injury','passing_yds',
        'passing_td','passing_att', 'passing_cmp',
        'receiving_yds','receiving_td','receiving_rec', 
        'receiving_tar', 'rushing_att','rushing_td', 
        'rushing_yds', 'FantasyPoints']]
        
        # Append the transformed DataFrame to the list
        all_transformed_dfs.append(df)
        print(f"Transformed data for {year} processed.")
    
    combined_df = pd.concat(all_transformed_dfs, ignore_index=True)

    # Save the combined DataFrame to a new CSV file in the output folder
    output_combined_file = os.path.join(output_folder, 'combined_player_transformed_data.csv')
    combined_df.to_csv(output_combined_file, index=False)
    print(f"All transformed data combined and saved to {output_combined_file}")
