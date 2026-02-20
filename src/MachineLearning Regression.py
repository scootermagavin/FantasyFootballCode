import pandas as pd

examp_df = pd.DataFrame({
 'ff_points_lag_1': [12, 20, 45, 68, 98], # lag means previous, 1 means this was last year, 2 would've meant 2 year ago, etc.
 'snapcount_lag_1': [10, 13, 14, 65, 70],
 'ff_points_next_year': [15, 40, 50, 55, 80]
})

import seaborn as sns; sns.set_style('whitegrid');

examp_df.plot(x='ff_points_lag_1', y='ff_points_next_year', kind='scatter');

sns.regplot(x=examp_df['ff_points_lag_1'], y=examp_df['ff_points_next_year']);

#bringing in already cleaned data
pd.set_option('display.max_columns', None)

df = pd.read_csv('https://gist.githubusercontent.com/fantasydatapros/304cd6d989cb9f9aa30221358b798e1a/raw/89c3ce4a9ee3e721468ca2051d7805f4b0d331df/regression_section_dataset.csv').iloc[:, 1:]

import numpy as np

df = df.groupby(['player_id', 'tm', 'player', 'pos', 'season'], as_index=False)\
    .agg({
    'offensive_snapcount': np.sum,
    'offensive_snapcount_percentage': np.mean,
    'passing_rating': np.mean,
    'passing_yds': np.sum,
    'passing_td': np.sum,
    'passing_att': np.sum,
    'receiving_yds': np.sum,
    'receiving_td': np.sum,
    'receiving_rec': np.sum,
    'receiving_tar': np.sum,
    'rushing_att': np.sum,
    'standard_fantasy_points': np.sum,
    'ppr_fantasy_points': np.sum,
    'half_ppr_fantasy_points': np.sum
})

df.loc[(df['season']>= 2012)]

pd.set_option('chained_assignment', None)

lag_features = ['rushing_att', 
                'receiving_tar', 
                'offensive_snapcount', 
                'offensive_snapcount_percentage',
                'ppr_fantasy_points',
                'passing_rating',
                'passing_att', 
                'passing_td']

for lag in range(1, 6):
    
    """
    We have not talked about shift before.
    Shift moves our data down by the number of rows we specify.
    
    pandas.DataFrame.shift documentation
    
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html
    """
    
    shifted = df.groupby('player_id').shift(lag)
    
    for column in lag_features:
        """
        Python f-strings are similar to using the format string method, although a bit cleaner
        
        Article on Python f-strings
        
        https://realpython.com/python-f-strings/
        """
        
        df[f'lag_{column}_{lag}'] = shifted[column]
        
df = df.fillna(-1)

df.corr()[['ppr_fantasy_points']]

"""9
Correlation matrix for just wide receivers.
"""

df.loc[df['pos'] == 'WR'].corr()[['ppr_fantasy_points']]

df.loc[df['pos'] == 'WR'].corr()[['ppr_fantasy_points']]

df.loc[df['pos'] == 'RB'].corr()[['ppr_fantasy_points']]

df.loc[df['pos'] == 'QB'].corr()[['ppr_fantasy_points']]

df.loc[df['pos'] == 'TE'].corr()[['ppr_fantasy_points']]

wr_df= df.loc[(df['pos'] =='WR') & (df['season'] < 2019)]

wr_df = wr_df.loc[wr_df['lag_offensive_snapcount_1'] > 50]

#this will be our feature matrix
X = wr_df[[
    'lag_receiving_tar_1', 'lag_offensive_snapcount_1', 'lag_ppr_fantasy_points_1'
]].values

y = wr_df['ppr_fantasy_points'].values

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# split up our data in to 20% testing, 80% training
# train_test_split documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# sklearn.linear_model.LinearRegression documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
lr = LinearRegression()

# train the algorithm
# the fit method documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.fit
lr.fit(X_train, y_train)

# predict method documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.predict
y_pred = lr.predict(X_test)

"""
A mean absolute error of 47 means our model was on average off by 47 fantasy points, or 3 points per game.
This is about what we'd expect from such a simple model.
"""
# mean_absolute_error documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
mean_absolute_error(y_pred, y_test)

pd.set_option('display.max_rows', None)

wr_df_pred = df.loc[
    (df['pos'] == 'WR') & (df['offensive_snapcount'] > 50) & (df['season'] == 2019), 
    ['player', 'receiving_tar', 'offensive_snapcount', 'ppr_fantasy_points']
]

wr_df_pred['predicted_2020'] = lr.predict(wr_df_pred[['receiving_tar', 'offensive_snapcount', 'ppr_fantasy_points']].values)

wr_df_pred.sort_values(by='predicted_2020', ascending=False).head(100)