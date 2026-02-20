#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 21:47:04 2022

@author: scotthoran
"""

from bs4 import BeautifulSoup as Soup
import pandas as pd
import requests
from pandas import DataFrame
import sqlite3
from os import path

NFL_DB_URL_Weekly = 'https://www.dailyfantasyfuel.com/nfl/projections/draftkings/2022-01-02/'

NFL_DB_Resp = requests.get(NFL_DB_URL_Weekly)

nfl_db_soup = Soup(NFL_DB_Resp.text)

nfl_db_tables = nfl_db_soup.find_all('table')

wk_17_tbl = nfl_db_tables[0]

wk_17_row = wk_17_tbl.find_all('tr') 

wk_17_first_row = wk_17_row[2]

wk_17_first_row.find_all('td')
[str(x.string) for x in wk_17_first_row.find_all('td')]

def parse_nfl_db_row(row):
    """
    take in tr tag and get the data out of it from a list of strings within
    Daily Fantasy Fuel's Projections table

    Parameters - rows of Daily Fantasy Fuel's data
    ----------
    row : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return [str(x.string) for x in row.find_all('td')]

def parse_nfl_db_header_row(row):
    """
    take in tr tag and get the data out of it from a list of strings within
    Daily Fantasy Fuel's Projections table

    Parameters - rows of Daily Fantasy Fuel's data
    ----------
    row : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return [str(x.string) for x in row.find_all('th')]

list_of_parsed_FF_rows = [parse_nfl_db_row(row) for row in wk_17_row[2:]]

nfl_db_df = DataFrame(list_of_parsed_FF_rows)

nfl_db_df.drop(columns=0, inplace= True)

nfl_db_df.columns = parse_nfl_db_header_row(wk_17_row[1])

nfl_db_df.drop(columns='None', inplace = True)

def scrape_fantasy_fuel_projected_lineup(date):
    """
    Need to have date format as YYYY-MM-DD

    Parameters
    ----------
    date : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    date_response = requests.get(f'https://www.dailyfantasyfuel.com/nfl/projections/draftkings/{date}/')
    date_soup = Soup(date_response.text)
    
    date_tables = date_soup.find_all('table')
    tables = date_tables[0]
    rows = tables.find_all('tr')
    
    list_of_parsed_rows = [parse_nfl_db_row(row) for row in wk_17_row[2:]]
    
    df = DataFrame(list_of_parsed_rows)
    df.drop(columns=0, inplace = True)
    
    df.columns = parse_nfl_db_header_row(rows[1])
    df.drop(columns='None', inplace = True)
    
    df['date'] = date
    
    return df

    df_wk_17_ff = scrape_fantasy_fuel_projected_lineup('2022-01-02')
    
    