#%%matplotlib inline
"""
Read more about pd.set_option here:
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
"""
import pandas as pd; pd.set_option('display.max_columns', None)
import seaborn as sns
from matplotlib import pyplot as plt #importing pyplot module from matplotlib. 

df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/2019.csv').iloc[:, 1:]
df['Usage/G'] = (df['PassingAtt'] + df['Tgt'] + df['RushingAtt']) / df['G'] 


"""
View the documentation for the info method here
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html
"""

#df.info(verbose=True)

pd.set_option('chained_assignment', None) # we don't care about overriding the original df

# you could also do rb_df = df.loc[df['Pos'] == 'RB'].copy()
rb_df = df.loc[df['Pos'] == 'RB']

# Usage is defined as Tgt + RushingAtt / games
rb_df['Usage/G'] = (rb_df['Tgt'] + rb_df['RushingAtt']) / rb_df['G'] 

"""
Here, we're just looking at the last column of our DataFrame here.
"""

#rb_df.iloc[:, -1:].head()

fantasy_scoring_weights = {
    'RushingYds': 0.1,
    'ReceivingYds': 0.1,
    'ReceivingTD': 6,
    'RushingTD': 6,
    'FumblesLost': -2,
    'Rec': 0.5, # adjust for PPR
    'PassingYds': .04,
    'PassingTD':4,
    'Int': -2
}

"""
Here, we are using apply with axis=1
This allows to map a function across an entire row, instead of across a column.
To learn more information about apply and setting axis=1,
check out the pandas documentation.
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
"""

def get_fantasy_points(row):
    
    g = row['G'] # access the Games column
    
    fantasy_points = sum([row[column]*weight for column, weight in fantasy_scoring_weights.items()])
    
    return fantasy_points / g

# create a FantasyPoints/G column. Syntax - dataframenam['new column name'] =
rb_df['FantasyPoints/G'] = rb_df.apply(get_fantasy_points, axis=1)
df['FantasyPoints/G'] = df.apply(get_fantasy_points, axis=1)
"""
The columns we need here are our Player, Tm, G, all relevant RB columns, Usage/G and FantasyPoints/G (which were just assigned)
"""

rb_df = rb_df[['Player', 'Tm', 'G', 'RushingAtt', 'Tgt'] + list(fantasy_scoring_weights.keys()) + ['Usage/G', 'FantasyPoints/G']]

sns.set_style('whitegrid') # setting style for visualizations, sets a baseline style here

"""
seaborn documentation for scatter plots:
https://seaborn.pydata.org/generated/seaborn.scatterplot.html
"""

# set figure size in inches
# https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.pyplot.figure.html
plt.figure(figsize=(8, 8))

sns.scatterplot(rb_df['RushingAtt'], rb_df['Tgt']); #takes an x & y to plot, semicolon removes excess wording

# mat plot lib version
#plt.scatter(rb_df['RushingAtt'], rb_df['Tgt']) Mat plot lib doesn't provide titles

#Regression plot (line of best fit around x & y data)

plt.figure(figsize=(12,10))

#shows a straight line fitting the data showing the relationship between the two. Seaborn won't give coefficients
sns.regplot(rb_df['Usage/G'], rb_df['FantasyPoints/G'])

plt.figure(figsize=(10,10))

#covering distribution plots to show data

#estimation of density on underlying distribution, rough estimation of distribution. Kernel Density Estimation
sns.kdeplot(rb_df['RushingAtt'])

#Follows a pareto distribution, 20% of causes relate to 80% of effects
#can combine with histogram

plt.figure(figsize=(10,10))

sns.distplot(rb_df['RushingAtt']) #turns above KDE plot into bar chart

plt.figure(figsize=(8,8))

#same thing but for targets

sns.distplot(rb_df['Tgt'], bins=30) #adding more bins

#plotting fantasy pts/g
plt.figure(figsize=(8,8))

sns.distplot(rb_df['FantasyPoints/G'], bins=30)

#bi-variate kernel density estimation plot. Plotting 2d density
#inner rings represent the most dense layers
fig, ax = plt.subplots(figsize=(8,8)) #takes in argument of amount of subplots you want on canvas. Default is 1

notable_players = ['Christian McCaffrey', 'Aaron Jones','Jamaal Williams', 'Mike Boone']

for player_name in notable_players:
    player = rb_df.loc[rb_df['Player'] == player_name]
    if not player.empty:
        targets = player['Tgt']
        rushes = player['RushingAtt']
        
        ax.annotate(player_name, xy=(rushes+2,targets+2), color='red', fontsize=12)
        ax.scatter(rushes, targets, color='red')

sns.kdeplot(rb_df['RushingAtt'], rb_df['Tgt'])


plt.figure(figsize=(10,10))
sns.jointplot(rb_df['RushingAtt'], rb_df['Tgt'], kind='kde')

#Residual Plots take independent and target variable, regress on them, x axis will be independent and y axis will be diff from line of best fit
#essentially shows the model's error
sns.set_style('dark')

sns.residplot(rb_df['Usage/G'], rb_df['FantasyPoints/G']) #here usage per gam is trying to explain fantasy points/g
#positive is model over-estimated the points, negative is under-estimated
#you want to try and see equal distribution across X and Y access
plt.title('Residual Plot')
plt.xlabel('Usage/G')
plt.ylabel('Residual')

#pairplot function to see paired scatterplot

rb_df_copy = rb_df[[
    'RushingAtt','RushingTD','FantasyPoints/G','Tgt']   
]

sns.pairplot(rb_df_copy); #pass in whole df

final_df = pd.DataFrame()

#note the curly braces in the string format. This allows the string to be updated with the format string method.
WEEKLY_BASE_URL = 'https://raw.githubusercontent.com/fantasydatapros/data/master/weekly/{year}/week{week_num}.csv'

year = 2019
#At each iteration its taking final_df and adding a new df to bottom
for week in range(1, 18):
    week_df = pd.read_csv(WEEKLY_BASE_URL.format(year=year, week_num=week)) # index_col=0 not necessary here. Data is properly formatted with Unnamed: 0 column
    week_df['Week'] = week 
    final_df = pd.concat([final_df, df])

#plotting Mahomes, Jackson, Wilson Fantasy points week by week

lamar = final_df.loc[final_df['Player'] == 'Lamar Jackson']
mahomes = final_df.loc[final_df['Player'] == 'Patrick Mahomes']
wilson = final_df.loc[final_df['Player'] == 'Russell Wilson']

sns.set_style('whitegrid')
plt.figure(figsize=(10,8))
plt.plot(wilson['Week'],wilson['StandardFantasyPoints'])
plt.plot(mahomes['Week'],mahomes['StandardFantasyPoints'])
plt.plot(lamar['Week'],lamar['StandardFantasyPoints'])
plt.legend(['Wilson','Mahomes','Lamar'])
plt.xlabel('Week')
plt.ylabel('Fantasy Points Scored')
plt.title('Wilson v Mahomes v Lamar - Week by Week');

#You can expand upon this to replicate with different code

#now we're moving to heat maps to visualize correlations between columns
#can calculate pearson which is r^2, or spearman which is helpful

lamar.corr()[['StandardFantasyPoints']] #values between -1 and 1. 1 means perfect correlation. Above .7 is strong. .6 to .8 is moderate effect

jameis_corr = final_df.loc[final_df['Player'] == 'Jameis Winston'].corr()[['StandardFantasyPoints']]
#can be helpful to see which players are more dependent on TD's or Throwing Interceptions
#We can pass Corr in to heatmap
plt.figure(figsize=(10,10))
sns.heatmap(jameis_corr, annot=True)

sns.lmplot(data=df, x='Usage/G', y='FantasyPoints/G', hue='Pos') #Good way to plot multiple regression lines based on a category

#Are TEs TD dependent?

corr = df.loc[df['Pos'] == 'TE'].corr()[['FantasyPoints/G']]
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True) #RecTD is highly correlated with Output but not more related than the others
#Usage actually may be a better determiner of output

#bringing in combine data
