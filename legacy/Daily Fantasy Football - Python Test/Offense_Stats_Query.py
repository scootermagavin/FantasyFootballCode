#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:04:48 2022

@author: scotthoran
"""

from bs4 import BeautifulSoup as Soup
import pandas as pd
import requests
from pandas import DataFrame

def parse_espn_row_pass(row):
    return [str(x.string) for x in row.find_all('td')]

def parse_espn_QB_pass(row):
    return [str(x.string) for x in row.find_all('td')[1]]

def scrape_espn_passing_wkly(weeknum):
    resp = requests.get(f'http://www.espn.com/nfl/weekly/leaders/_/week/{weeknum}/type/passing')
    resp_soup = Soup(resp.text)
    tables = resp_soup.find_all('table')
    
    passing_tables = tables[6]
    passing_row = passing_tables.find_all('tr')
    
    parsed_rows = [parse_espn_row_pass(row) for row in passing_row[2:]]
    parsed_QB   = [parse_espn_QB_pass(row) for row in passing_row[2:]]
    
    df = DataFrame(parsed_rows)
    
    df_header_list = parse_espn_row_pass(passing_row[1])
    
    df.columns = df_header_list
    df.columns = [x.lower() for x in df.columns]    
    df_header = DataFrame(parsed_QB)
    df_final = df.join(df_header)
    df_final.drop('player',axis=1,inplace=True)
    df_final.rename(columns={0:'player',1:'position'}, inplace=True)
    df_final.drop('position',axis=1,inplace=True)
    df_final.drop('result',axis=1,inplace=True)
    df_final['WeekNum'] = weeknum
    return df_final

Weeks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

df_ultimate = DataFrame(columns = ['rk','team','comp','att','yds','td','int','sack','fum','rat','player'])

for num in Weeks:
    df_weekly = scrape_espn_passing_wkly(num)
    df_ultimate = pd.concat([df_ultimate,df_weekly])
    
def parse_espn_row_rush(row):
    return [str(x.string) for x in row.find_all('td')]

def parse_espn_RB(row):
    return [str(x.string) for x in row.find_all('td')[1]]

def scrape_espn_rushing_wkly(weeknum):
    resp = requests.get(f'http://www.espn.com/nfl/weekly/leaders/_/week/{weeknum}/type/rushing')
    resp_soup = Soup(resp.text)
    tables = resp_soup.find_all('table')
    
    rushing_tables = tables[6]
    rushing_row = rushing_tables.find_all('tr')
    
    parsed_rows = [parse_espn_row_rush(row) for row in rushing_row[2:]]
    parsed_RB   = [parse_espn_RB(row) for row in rushing_row[2:]]
    
    df = DataFrame(parsed_rows)
    
    df_header_list = parse_espn_row_rush(rushing_row[1])
    
    df.columns = df_header_list
    df.columns = [x.lower() for x in df.columns]    
    df_header = DataFrame(parsed_RB)
    df_final = df.join(df_header)
    df_final.drop('player',axis=1,inplace=True)
    df_final.rename(columns={0:'player',1:'position'}, inplace=True)
    df_final.drop('position',axis=1,inplace=True)
    df_final.drop('result',axis=1,inplace=True)
    df_final['WeekNum'] = weeknum
    return df_final

df_ultimate_rb = DataFrame(columns = ['rk','team','car','avg','yds','td','lng','fum','player'])

for num in Weeks:
    df_weekly_rb = scrape_espn_rushing_wkly(num)
    df_ultimate_rb = pd.concat([df_ultimate_rb,df_weekly_rb])
    
def parse_espn_row_receive(row):
    return [str(x.string) for x in row.find_all('td')]

def parse_espn_WR(row):
    return [str(x.string) for x in row.find_all('td')[1]]

def scrape_espn_receiving_wkly(weeknum):
    resp = requests.get(f'http://www.espn.com/nfl/weekly/leaders/_/week/{weeknum}/type/receiving')
    resp_soup = Soup(resp.text)
    tables = resp_soup.find_all('table')
    
    receiving_tables = tables[6]
    receiving_row = receiving_tables.find_all('tr')
    
    parsed_rows = [parse_espn_row_receive(row) for row in receiving_row[2:]]
    parsed_WR   = [parse_espn_WR(row) for row in receiving_row[2:]]
    
    df = DataFrame(parsed_rows)
    
    df_header_list = parse_espn_row_receive(receiving_row[1])
    
    df.columns = df_header_list
    df.columns = [x.lower() for x in df.columns]    
    df_header = DataFrame(parsed_WR)
    df_final = df.join(df_header)
    df_final.drop('player',axis=1,inplace=True)
    df_final.rename(columns={0:'player',1:'position'}, inplace=True)
    df_final.drop('position',axis=1,inplace=True)
    df_final.drop('result',axis=1,inplace=True)
    df_final['WeekNum'] = weeknum
    return df_final

df_ultimate_wr = DataFrame(columns = ['rk','team','rec','yds','avg','td','lng','fum','player'])

for num in Weeks:
    df_weekly_wr = scrape_espn_receiving_wkly(num)
    df_ultimate_wr = pd.concat([df_ultimate_wr,df_weekly_wr])    
    
df_QB_RB = pd.concat([df_ultimate,df_ultimate_rb],axis = 1)

resp_fd = requests.get('https://fantasydata.com/nfl/fantasy-football-leaders?position=7&season=2021&seasontype=1&scope=2&subscope=1&startweek=1&endweek=17&aggregatescope=1&range=3')

resp_fd_soup = Soup(resp_fd.text)

fd_tables = resp_fd_soup.find_all('table')

