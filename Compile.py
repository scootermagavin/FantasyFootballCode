#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:30:09 2022

@author: scotthoran
"""

import pandas as pd

positions = ['rb', 'wr', 'qb', 'te']

dfs = []

for position in positions:
    df = pd.read_csv(f'data/{position}.csv')
    df.drop(df.index[0], inplace=True)
    df['POS'] = position.upper()
    if position == 'rb':
        df = df.rename({
            'ATT': 'RushingAtt',
            'YDS': 'RushingYds',
            'TDS': 'RushingTD',
            'YDS.1': 'ReceivingYds',
            'TDS.1': 'ReceivingTD', 
            'REC': 'Receptions'
        }, axis=1)
    elif position == 'qb':
        df = df.rename({
            'ATT': 'PassingAtt',
            'YDS': 'PassingYds',
            'TDS': 'PassingTD',
            'ATT.1': 'RushingAtt',
            'YDS.1': 'RushingYds',
            'TDS.1': 'RushingTD'
        }, axis=1)
    elif position == 'wr':
        df = df.rename({
            'ATT': 'RushingAtt',
            'YDS': 'ReceivingYds',
            'TDS': 'ReceivingTD',
            'YDS.1': 'RushingYds',
            'TDS.1': 'RushingTD',
            'REC': 'Receptions'
        }, axis=1)
    elif position == 'te':
        df = df.rename({
            'YDS': 'ReceivingYds',
            'TDS': 'ReceivingTD',
            'REC': 'Receptions'
        }, axis=1)
    #df = df.drop('FPTS', axis=1)
    dfs.append(df)

df = pd.concat(dfs)
df = df.fillna(0)

df = df.loc[:, df.columns[0:2].tolist()+['POS']+df.columns[2:].tolist()]
df = df.loc[:,~df.columns.duplicated()]

df.to_csv('data/all_compiled.csv')