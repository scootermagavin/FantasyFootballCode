#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:58:37 2022

@author: scotthoran
"""

import numpy as np
import matplotlib.pyplot as plt
from math import floor
import pandas as pd
from random import sample

#standardize actual fp

def reduceForProjectionAnalysis(file):
    file = file[file['injury_status'] != 'O']
    file = file[file['injury_status'] != 'Q']
    file['FantasyFuelPPGProj'] = file['ppg_projection']
    file['FantasyFuelValueProj'] = file['value_projection']
    file['DFN_PPGProj'] = file['Proj FP']
    file['DFN_ValueProj'] = file['Proj Val']
    
    cleaned_file = pd.DataFrame()
    cleaned_file['Player Name'] = file['Player Name']
    cleaned_file['Actual FP'] = file['Actual FP']
    cleaned_file['team'] = file['team']
    cleaned_file['Pos'] = file['Pos']
    cleaned_file['Salary'] = file['Salary']
    cleaned_file['DFN_MinProj'] = file['Proj Min']
    cleaned_file['Avg Proj'] = (file['DFN_PPGProj']+file['FantasyFuelPPGProj'])/2
    cleaned_file['Avg Value Proj'] = ((file['DFN_ValueProj']+file['FantasyFuelValueProj'])/2)
    cleaned_file['Avg Skewed Min'] = ((file['L2 Min']+file['L5 Min']+file['S Min'])/3)
    cleaned_file['Avg Skewed FGA'] = ((file['L2 FGA']+file['L5 FGA']+file['S FGA'])/3)
    cleaned_file['Avg Skewed FP'] = ((file['L5 FP']+file['S FP']+file['Ceil FP']+file['Floor FP']+cleaned_file['Avg Proj'])/5)
    cleaned_file['Proj Min Enhanced'] = (cleaned_file['Avg Proj'] / cleaned_file['Avg Skewed Min'] ) * cleaned_file['DFN_MinProj']
    cleaned_file = cleaned_file.replace([np.inf, -np.inf], 0)
    cleaned_file = cleaned_file.fillna(0)
    return cleaned_file

##Helper Functions
def verifyLineup(lineup):
    enough_represented_teams = len(set(lineup['team'].tolist()))>=2
    under_salary_cap = lineup['Salary'].sum() <= 50000
    all_unique_players = len(set(lineup['Player Name'].tolist())) == 8
    if enough_represented_teams and under_salary_cap and all_unique_players:
        return True
    else:
        return False
    
def createRandomPopulation(point_guards, shooting_guards, small_forwards, power_forwards, guards, forwards, centers, util, limit):
    counter = 0
    lineups = []
    while(counter<limit):
        lineup = pd.DataFrame()
        lineup = lineup.append(point_guards.sample(n=1))
        lineup = lineup.append(shooting_guards.sample(n=1))
        lineup = lineup.append(small_forwards.sample(n=1))
        lineup = lineup.append(power_forwards.sample(n=1))
        lineup = lineup.append(guards.sample(n=1))
        lineup = lineup.append(forwards.sample(n=1))
        lineup = lineup.append(centers.sample(n=1))
        lineup = lineup.append(util.sample(n=1))
        if(verifyLineup(lineup)):
            lineups.append(lineup)
            counter = counter + 1
    return lineups

def randomlyCreateLineups(file):
    population_size = 10
    point_guards = file.loc[file['Pos'].str.contains("PG")]
    shooting_guards = file.loc[file['Pos'].str.contains("SG")]
    small_forwards = file.loc[file['Pos'].str.contains("SF")]
    power_forwards = file.loc[file['Pos'].str.contains("PF")]
    guards = file.loc[file['Pos'].str.contains("G")]
    forwards = file.loc[file['Pos'].str.contains("F")]
    centers = file.loc[file['Pos'].str.contains("C")]
    util = file
    population = createRandomPopulation(point_guards, shooting_guards, 
                                               small_forwards, power_forwards, 
                                               guards, forwards, centers, util, population_size)  
       
    return population


dates_all = [
    ['3_11', '2020-03-11'],
    ['3_10', '2020-03-10'],
    ['3_9', '2020-03-09'],
    ['3_8', '2020-03-08'],
    ['3_7', '2020-03-07'],
    ['3_6', '2020-03-06'],
    ['3_5', '2020-03-05'],
    ['3_4', '2020-03-04'],
    ['3_3', '2020-03-03'],
    ['3_2', '2020-03-02'],
    ['3_1', '2020-03-01'],
    ['2_29', '2020-02-29'],
    ['2_28', '2020-02-28'],
    ['2_27', '2020-02-27'],
    ['2_26', '2020-02-26'],
    ['2_25', '2020-02-25'],
    ['2_24', '2020-02-24'],
    ['2_23', '2020-02-23'],
    ['2_22', '2020-02-22'],
    ['2_21', '2020-02-21'],
    ['2_20', '2020-02-20'],
    ['2_13', '2020-02-13'],
    ['2_12', '2020-02-12'],
    ['2_11', '2020-02-11'],
    ['2_10', '2020-02-10'],
    ['2_9', '2020-02-09'],
    ['2_8', '2020-02-08'],
    ['2_7', '2020-02-07'],
    ['2_6', '2020-02-06'],
    ['2_5', '2020-02-05'],
    ['2_4', '2020-02-04'],
    ['2_3', '2020-02-03'],
    ['2_2', '2020-02-02'],
    ['2_1', '2020-02-01'],
    ['1_31', '2020-01-31'],
    ['1_30', '2020-01-30'],
    ['1_29', '2020-01-29'],
    ['1_28', '2020-01-28'],
    ['1_27', '2020-01-27'],
    ['1_26', '2020-01-26'],
    ['1_25', '2020-01-25'],
    ['1_24', '2020-01-24'],
    ['1_23', '2020-01-23'],
    ['1_22', '2020-01-22'],
    ['1_20', '2020-01-20'],
    ['1_19', '2020-01-19'],
    ['1_18', '2020-01-18'],
    ['1_17', '2020-01-17'],
    ['1_16', '2020-01-16'],
    ['1_15', '2020-01-15'],
    ['1_14', '2020-01-14'],
    ['1_13', '2020-01-13'],
    ['1_12', '2020-01-12'],
    ['1_11', '2020-01-11'],
    ['1_10', '2020-01-10'],
    ['1_9', '2020-01-09'],
    ['1_8', '2020-01-08'],
    ['1_7', '2020-01-07'],
    ['1_6', '2020-01-06'],
    ['1_5', '2020-01-05'],
    ['1_4', '2020-01-04'],
    ['1_3', '2020-01-03'],
    ['1_2', '2020-01-02'],
    ['1_1', '2020-01-01'],
    ['12_31', '2019-12-31'],
    ['12_30', '2019-12-30'],
    ['12_29', '2019-12-29'],
    ['12_28', '2019-12-28'],
    ['12_27', '2019-12-27'],
    ['12_26', '2019-12-26'],
    ['12_25', '2019-12-25'],
    ['12_23', '2019-12-23'],
    ['12_22', '2019-12-22'],
    ['12_21', '2019-12-21'],
    ['12_20', '2019-12-20'],
    ['12_19', '2019-12-19'],
    ['12_18', '2019-12-18'],
    ['12_17', '2019-12-17'],
    ['12_16', '2019-12-16'],
    ['12_15', '2019-12-15'],
    ['12_14', '2019-12-14'],
    ['12_13', '2019-12-13'],
    ['12_12', '2019-12-12'],
    ['12_11', '2019-12-11'],
    ['12_10', '2019-12-10'],
    ['12_9', '2019-12-09'],
    ['12_8', '2019-12-08'],
    ['12_7', '2019-12-07'],
    ['12_6', '2019-12-06'],
    ['12_5', '2019-12-05'],
    ['12_4', '2019-12-04'],
    ['12_3', '2019-12-03'],
    ['12_2', '2019-12-02'],
    ['12_1', '2019-12-01'],
    ['11_30', '2019-11-30'],
    ['11_29', '2019-11-29'],
    ['11_27', '2019-11-27'],
    ['11_26', '2019-11-26'],
    ['11_25', '2019-11-25'],
    ['11_24', '2019-11-24'],
    ['11_23', '2019-11-23'],
    ['11_22', '2019-11-22'],
    ['11_21', '2019-11-21'],
    ['11_20', '2019-11-20'],
    ['11_19', '2019-11-19'],
    ['11_18', '2019-11-18'],
    ['11_17', '2019-11-17'],
    ['11_16', '2019-11-16'],
    ['11_15', '2019-11-15'],
    ['11_14', '2019-11-14'],
    ['11_13', '2019-11-13'],
    ['11_12', '2019-11-12'],
    ['11_11', '2019-11-11'],
    ['11_10', '2019-11-10'],
    ['11_9', '2019-11-09'],
    ['11_8', '2019-11-08'],
    ['11_7', '2019-11-07'],
    ['11_6', '2019-11-06'],
    ['11_5', '2019-11-05'],
    ['11_4', '2019-11-04'],
    ['11_3', '2019-11-03'],
    ['11_2', '2019-11-02'],
    ['11_1', '2019-11-01'],
    ['10_31', '2019-10-31'],
    ['10_30', '2019-10-30'],
    ['10_29', '2019-10-29'],
    ['10_28', '2019-10-28'],
    ['10_27', '2019-10-27'],
    ['10_26', '2019-10-26'],
    ['10_25', '2019-10-25'],
    ['10_24', '2019-10-24'],
    ['10_23', '2019-10-23'],
    ['10_22', '2019-10-22'],   
]


def resetIndicies(file):
    for lineup in file:
        lineup.reset_index(drop=True, inplace=True)
    return file

def saveBestLineups_toCSV(file, date):
    key = np.arange(len(file))
    file_name = 'RandomlyCreatedLineups/{}'.format(date)
    new_file = pd.concat(file, keys=key, names=['Lineup Num'])
    new_file.to_csv(file_name)

def createRandomLineups():
    for game_day in dates_all:
        file_name = 'HistoricalData_Merged/{}'.format(game_day[1])
        file = pd.read_csv(file_name)
        cleaned_file = reduceForProjectionAnalysis(file)
        random_population = randomlyCreateLineups(cleaned_file)
        saveBestLineups_toCSV(resetIndicies(random_population), game_day[1])
        
createRandomLineups()