#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 06:47:18 2022

@author: scotthoran
"""

#2021 season data and future data available here https://www.fantasyfootballdatapros.com/course/section/6

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/74b84c5fb2371b954b52b4f67ae5220930d57861/fantasypros/fp_projections.csv')
df= df.iloc[:,1:]
#done because first column of .csv was bad format

#individual league scoring attributes
scoring_weights = {
    'receptions': 1, # PPR
    'receiving_yds': 0.1,
    'receiving_td': 6,
    'FL': -2, #fumbles lost
    'rushing_yds': 0.1,
    'rushing_td': 6,
    'passing_yds': 0.04,
    'passing_td': 4,
    'int': -2
}

# make sure you add paranthesis around the right side of this expression
# overriding the fantasy points column with our own league settings
# Syntax for new dataframe column based off other columns: 
# df['NewColumnName'] = df['oldcolumn1']+df['oldcolumn2']....

df['FantasyPoints'] = (
    df['Receptions']*scoring_weights['receptions'] + df['ReceivingYds']*scoring_weights['receiving_yds'] + \
    df['ReceivingTD']*scoring_weights['receiving_td'] + df['FL']*scoring_weights['FL'] + \
    df['RushingYds']*scoring_weights['rushing_yds'] + df['RushingTD']*scoring_weights['rushing_td'] + \
    df['PassingYds']*scoring_weights['passing_yds'] + df['PassingTD']*scoring_weights['passing_td'] + \
    df['Int']*scoring_weights['int'] )

    # mask our dataframe based off a position
"""
.loc is a way of getting back specified cross sections of your dataframe.
The syntax is as follows:
new_df = old_df.loc[row_indexer, column_indexer]
Where row_indexer can take the form of a boolean indexer.
For example, df['Pos'] == 'RB'
or, df['RushingAtt'] > 20
or, df['Pos'].isin(['QB', 'WR', 'RB', TE]) 
# check if a player's position is a skill position
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html 
# docs on loc
https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html # docs on indexing
"""

rb_df = df.loc[df['Pos'] == 'RB']

rb_df.head()

base_columns = ['Player', 'Team', 'Pos']
rushing_columns = ['FantasyPoints', 'Receptions', 'ReceivingYds', 'ReceivingTD', 'RushingAtt', 'RushingYds', 'RushingTD', 'FL']

"""
Here, we can mask (what we are doing in the row indexer) and filter (what we are doing in the column indexer)
all in one line. Pass in (the boolean indexer, columns you'd like to keep) as a tuple.
Also recall that lists can be concatenated together.
"""
rb_df = df.loc[(df['Pos'] == 'RB', base_columns + rushing_columns)]

"""
The sort_values method of a DataFrame allows us sort our table by a given column.
The 'by' parameter of the function here is a required argument, and it should be the name of 
one of the columns in your table.
The 'ascending' argument is optional. If you want to sort your table from largest to smallest, set
ascending = False to sort in descending order. The object we get back from the sort_values function
is also a pandas DataFrame, and so we can chain methods as we do below with sort_values and head.
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
"""

# sort RBs by RushingYds in descending order and get us back the top 15 rows.
rb_df.sort_values(by='RushingYds', ascending=False).head(15)

"""
We can use the describe method to get summary/descriptive statistics about our DataFrame extremely quickly
We can also use transpose to switch the columns and index.
# describe docs
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
# transpose documentation
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transpose.html
"""

# get us back descriptive stats about our rb_df and transpose the DataFrame.
rb_df_describe_transpose = rb_df.describe().transpose()

rb_df['RushingTDRank'] = rb_df['RushingTD'].rank(ascending=False)

#import seaborn as sns
#sns.set_style('whitegrid')
#sns.distplot(rb_df['RushingAtt'])

henry = rb_df.loc[rb_df['Player'] == 'Derrick Henry']
# we can grab in
henry = henry.transpose() # transpose the DataFrame
# this is how we rename the index.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.rename.html
henry.index = henry.index.rename('Category') 

"""
This is how we rename the index in pandas. Note that length of our original columns was 1, and the list you 
set equal to df.columns must have the same length as the original column index. Otherwise, a ValueError will
be thrown.
"""
henry.columns = ['Value'] 
henry


#Adding in the ADP Data

adp_df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/fantasypros/adp/PPR_ADP.csv', index_col=0) # set index col = 0 to set the range index as our dataframes index
adp_df['ADP Rank'] = adp_df['AVG'].rank()

#goal is to find the last player drafted at each position at the 100 point in the draft
adp_df_cutoff = adp_df[:100]
# shape get's us back a tuple with the number of rows, and number of columns
# you can also use the Python built-in function len() to find the number of rows.

# print(len(adp_dp_cutoff))
#adp_df_cutoff.shape
# initialize an empty dictionary.
# this is where we are going to save our replacement players

replacement_players = {
    'RB': '',
    'QB': '',
    'WR': '',
    'TE': ''
}

"""
We can iterate over our DataFrame using the iterrows method.
It's similar to the items method for dictionary looping.
Instead of key, values in dict.items(), we get back the index, row in df.items()
Here, we're using _ as the placeholder value for the index. Whenever you do not use a variable when looping,
you should use _ as a placeholder to tell yourself later and other people reading your code that this is a 
dummy variable. This is good style.
Using iterrows, we can iterate over our DataFrame, and get access to each row's column values.
We can access these column value much like we would values in a dictionary, using the [] notation.
Here, we are constantly re-setting the keys in the dict we instantiated above with the most recent player from 
our loop, if their position is in replacement_players keys (in other words, if they are a WR, RB, TE, or qb)
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html
"""

for _, row in adp_df_cutoff.iterrows():
    
    position = row['POS'] # extract out the position and player value from each row as we loop through it
    player = row['PLAYER']
    
    if position in replacement_players: # if the position is in the dict's keys
        replacement_players[position] = player # set that player as the replacement player

"""
This is how we filter out columns in pandas without using loc.
The syntax is as follows
df = df[columns_wed_like_to_keep_in_list_form]
"""

df = df[['Player','Pos','Team','FantasyPoints']]

replacement_values = {} #initialize an empty dictionary

for position, player_name in replacement_players.items():
    
    player = df.loc[df['Player'] == player_name]
    
    # because this is a series object we get back, we need to use the tolist method
    # to get back the series as a list. The list object is of length 1, and the 1 item has the value we need.
    # we tack on a [0] to get the value we need.
    
    replacement_values[position] = player['FantasyPoints'].tolist()[0]

"""
the isin method lets us check if a value is in a list
and can be passed as a boolean indexer / row filter / mask
here, we want to filter out all those rows who's position column is not in
['QB', 'RB', 'WR', 'TE']
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html
""" 

# this will be explained in the next chapter
pd.set_option('chained_assignment', None)

df = df.loc[df['Pos'].isin(['QB', 'RB', 'WR', 'TE'])]

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
# applied lambda function across every row in the table and added 
df['VOR'] = df.apply(
    lambda row: row['FantasyPoints'] - replacement_values.get(row['Pos']), axis=1
)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
pd.set_option('display.max_rows', None) # turn off truncation of rows setting inherent to pandas

df['VOR Rank'] = df['VOR'].rank(ascending=False)
"""
the pandas groupby method allows us to groupby a specific column, called "splitting",
then apply a summary function over to each group. We can split this up by column as well, by tacking
on ['ColumName'] after grouping and before applying the summary function.
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
"""

df.groupby('Pos')['VOR'].describe()

# standard score example. Notice axis is not set to 1 as we are applying a function across a column here.
# df['VOR'] = df['VOR'].apply(lambda x: (x - df['VOR'].mean()) / df['VOR'].std())

df['VOR'] = df['VOR'].apply(lambda x: (x - df['VOR'].min()) / (df['VOR'].max() - df['VOR'].min()))

df = df.sort_values(by='VOR Rank')

import seaborn as sns 
# calculating how many players are in our draft pool.
num_teams = 12
num_spots = 16 # 1 QB, 2RB, 2WR, 1TE, 1FLEX, 1K, 1DST, 7 BENCH
draft_pool = num_teams * num_spots

df_copy = df[:draft_pool]

sns.boxplot(x=df_copy['Pos'], y=df_copy['VOR']);

# let's rename our VOR column to just Value.
# remember, to make a change to our DataFrame, you set it equal to itself + some modifcation
# we can use the rename method here to help us do that
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html

df = df.rename({
    'VOR': 'Value',
    'VOR Rank': 'Value Rank'
}, axis=1) # axis = 1 means make the change along the column axis.

adp_df = adp_df.rename({
    'PLAYER': 'Player',
    'POS': 'Pos',
    'AVG': 'Average ADP',
    'ADP RANK': 'ADP Rank'
}, axis=1) # let's rename some columns first.

"""
the merge function allows us to combine DataFrames together column wise on common columns.
Here, we are left joining. Which means any entries that exist in the right table (adp_df) but
do not exist in the left table (df) get dropped from the final table.
We want to join the two DataFrames together where the Player and Pos columns match up.
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
Also, you may want to take a look at join, which is similar
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html
"""

final_df = df.merge(adp_df, how='left', on=['Player', 'Pos'])
# let's calculate the difference between our value rank and adp rank
final_df['Diff in ADP and Value'] = final_df['ADP Rank'] - final_df['Value Rank']

#We want to watch out for those players who have positive Diff in ADP/Value and avoid those who don't
draft_pool = final_df.sort_values(by='ADP Rank')[:196]

rb_draft_pool = draft_pool.loc[draft_pool['Pos'] == 'RB']
qb_draft_pool = draft_pool.loc[draft_pool['Pos'] == 'QB']
wr_draft_pool = draft_pool.loc[draft_pool['Pos'] == 'WR']
te_draft_pool = draft_pool.loc[draft_pool['Pos'] == 'TE']


