#this will try to address the benefit of stacking players from same team
#looking at pairwise correlations

import pandas as pd
import time

df = pd.DataFrame()

start = time.time()

WEEKLY_BASE_URL = "https://raw.githubusercontent.com/fantasydatapros/data/master/weekly/{year}/week{week}.csv"

year = 2019

for week in range(1,18):
    weekly_df = pd.read_csv(WEEKLY_BASE_URL.format(year=year,week=week))
    weekly_df['Week'] = week 
    df =pd.concat([df,weekly_df])
    
    seconds = time.time() - start
    
    print(f'Added data for week {week}, {seconds} seconds have passed')

#can use to_csv to save this out somewhere

df = df.replace({
    'Pos':{
        'HB': 'RB',
        'WR/RS':'WR',
        'WR/PS':'WR',
        'FB/TE':'TE',
        'FB/HB':'RB',
        'WR/PR':'WR'
    }
})

skill_positions = ['QB','WR','TE','RB']

df = df.loc[df['Pos'].isin(skill_positions)]

import numpy as np

columns = ['Player', 'Tm','Pos', 'Week','PPRFantasyPoints']

new_df = df[columns]

new_df = new_df.groupby(['Player', 'Tm','Pos'], as_index=False).agg({
    'PPRFantasyPoints': np.mean
})

position_map = {
    'QB': 1,
    'RB': 2,
    'WR': 3,
    'TE': 2
}

def get_top_n_player_at_each_position(df, pos, n):
    df = df.loc[df['Pos'] == pos]
    """
    For each group, grab the nlargest FantasyPoints for the position
    given the n argument set.
    
    For example, to find the WR3 of a team:
    
    Set n=3 and pos=WR
    
    The nlargest function will get us back the top 3WRs in terms of FantasyPoints
    
    To grab the WR3, get the smallest FantasyPoints output from the group (using the min function)
    
    """   
    return df.groupby('Tm', as_index=False).apply(
        lambda x: x.nlargest(n, ['PPRFantasyPoints']).min()
    )

corr_df = pd.DataFrame(columns=columns)

"""
Iterate over every position. For each position, also iterate over the range of 1 -> n + 1
For example, in the case of WR, we are going to iterate from WR1 -> WR3 and concatenate the values column wise
"""

for pos, n_spots in position_map.items():
    
    for n in range(1, n_spots + 1):
        
        pos_df = get_top_n_player_at_each_position(new_df, pos, n)
        
        """
        Rename the 
        """
        pos_df = pos_df.rename({'PPRFantasyPoints': f'{pos}{n}'}, axis=1)
        
        """
        
        To concatenate column wise, you can use axis=1.
        
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
        
        """
        
        corr_df = pd.concat([corr_df, pos_df], axis=1)
        
corr_df = corr_df.dropna(axis=1) # Setting axis=1 drops those columns with NA values.
corr_df = corr_df.drop(['Pos', 'Player', 'Tm'], axis=1)

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style('whitegrid')

plt.figure(figsize=(10,7))

"""
sns.diverging palette gets us back an object that can alter the colormap of our visualization.
"""
sns.heatmap(corr_df.corr(), annot=True, cmap=sns.diverging_palette(0, 250));


