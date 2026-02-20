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

combined_df = pd.read_csv('/Users/scotthoran/Library/CloudStorage/OneDrive-Personal/Documents/Daily Fantasy - Active/DFS Python Build/Transformation/combined_player_transformed_data.csv')
combined_df['yard_per_completion'] = combined_df['passing_yds'] / combined_df['passing_cmp']

columns_to_clean = ['passing_yds', 'passing_td',
       'passing_att', 'passing_cmp', 'receiving_yds', 'receiving_td',
       'receiving_rec', 'receiving_tar', 'rushing_att', 'rushing_td',
       'rushing_yds', 'yard_per_completion','FantasyPoints']
combined_df[columns_to_clean] = combined_df[columns_to_clean].fillna(0)


weekly_columns = ["player_id", "name", "position_id", "season", 'week', 'passing_cmp','passing_yds','yard_per_completion','receiving_rec','receiving_yds','receiving_tar','rushing_att','rushing_yds','FantasyPoints']
regular_season_df = combined_df[weekly_columns]
