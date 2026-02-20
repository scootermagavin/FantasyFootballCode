#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:36:38 2022

@author: scotthoran
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import floor
import pandas as pd
from random import sample

drop_attribs = [
        "game_date",
        'ppg_projection', 
        'value_projection', 
        'Proj FP', 
        'Proj Min', 
        'Proj Val',
        "position",
        "salary", 
]

#Renames columns in the historical merged data files
def reduceForProjectionAnalysis(file):
    file['FantasyFuelPPGProj'] = file['ppg_projection']
    file['FantasyFuelValueProj'] = file['value_projection']
    file['DFN_PPGProj'] = file['Proj FP']
    file['DFN_MinProj'] = file['Proj Min']
    file['DFN_ValueProj'] = file['Proj Val']
    file['Avg Proj'] = (file['DFN_PPGProj']+file['FantasyFuelPPGProj'])/2
    file['Avg Value Proj'] = ((file['DFN_ValueProj']+file['FantasyFuelValueProj'])/2)
    file['Avg Skewed Min'] = ((file['L2 Min']+file['L5 Min']+file['S Min'])/3)
    file['Avg Skewed FGA'] = ((file['L2 FGA']+file['L5 FGA']+file['S FGA'])/3)
    file['Avg Skewed FP'] = ((file['L5 FP']+file['S FP']+file['Ceil FP']+file['Floor FP']+file['Avg Proj'])/5)
    file['Avg L5 PPG'] = ((file['L5_ppg_floor']+file['L5_ppg_avg']+file['L5_ppg_max'])/3)
    file['Proj Min Enhanced'] = (file['Avg Proj'] / file['Avg Skewed Min'] ) * file['Proj Min']
    file.drop(drop_attribs, axis=1, inplace=True)
    file = file.loc[file['Actual FP'] > 0]
    return file



##Helper Functions

def verifyLineup(lineup):
    enough_represented_teams = len(set(lineup['team'].tolist()))>=2 ##Makes sure more than 2 teams are represented in the lineup
    under_salary_cap = lineup['Salary'].sum() <= 50000 ##ensures lineup is below salary cap
    all_unique_players = len(set(lineup['Player Name'].tolist())) == 8 #Checks to make sure there are no repeat player names
    if enough_represented_teams and under_salary_cap and all_unique_players:
        return True
    else:
        return False
    
def createRandomPopulation(point_guards, shooting_guards, small_forwards, power_forwards, guards, forwards, centers, util, limit):
    counter = 0
    lineups = [] ##Creates a population of randomized lineups
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

##unsure, I guess this creates children lineups to test model

def mate(_parents):
    parents = pd.concat(sample(_parents, 10))
    point_guards = parents.loc[parents['Pos'].str.contains("PG")]
    shooting_guards = parents.loc[parents['Pos'].str.contains("SG")]
    small_forwards = parents.loc[parents['Pos'].str.contains("SF")]
    power_forwards = parents.loc[parents['Pos'].str.contains("PF")]
    guards = parents.loc[parents['Pos'].str.contains("G")]
    forwards = parents.loc[parents['Pos'].str.contains("F")]
    centers = parents.loc[parents['Pos'].str.contains("C")]
    util = parents
    
    children = []
    while(len(children) < 10):
        child_lineup = _parents[0].append(_parents[1])
        while(not verifyLineup(child_lineup)):
            child_lineup = pd.DataFrame()
            child_lineup = child_lineup.append(point_guards.sample(n=1))
            child_lineup = child_lineup.append(shooting_guards.sample(n=1))
            child_lineup = child_lineup.append(small_forwards.sample(n=1))
            child_lineup = child_lineup.append(power_forwards.sample(n=1))
            child_lineup = child_lineup.append(guards.sample(n=1))
            child_lineup = child_lineup.append(forwards.sample(n=1))
            child_lineup = child_lineup.append(centers.sample(n=1))
            child_lineup = child_lineup.append(util.sample(n=1))
        children.append(child_lineup)
    return children

##Sorts Population by actuals and eliminates duplicates

def sortFitest(population): 
    population.sort(key=actualPointSum, reverse=True)
    population = eliminateDuplicates(population)
    return population 

#Returns actual points

def actualPointSum(lineup):
    return lineup['Actual FP'].sum() 

def eliminateDuplicates(population):
    unique_lineups = set()
    unique_population = []
    for lineup in population:
        lineup_set = set()
        for player in lineup['Player Name'].tolist():
            lineup_set.add(player)
            
        if(not lineup_set.issubset(unique_lineups)):  
            unique_population.append(lineup)
            unique_lineups = unique_lineups.union(lineup_set)
    return unique_population

##selects the most fitting population

def performSelection(population):
    selected_population = population[:1]
    selected_population.extend(sample(population[1:10], 5))
    selected_population.extend(sample(population[10:20], 4))
    selected_population.extend(sample(population, 9))
    return selected_population

def performCrossover(population):
    children = []
    parents = population
    children = mate(parents) 
    return children
        
##Create generational population, 
        
def createNextGeneration(old_population):
    selected_population = performSelection(old_population)
    children = performCrossover(selected_population)
    new_generation = old_population[:len(old_population)-10]
    new_generation.extend(children)
    return new_generation

##Creates historical lineups algo

def geneticAlgorithmForHistorialBestLineups(file):
    population_size = 50
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
    counter = 0
    while(counter<100):
        currentPopulation = sortFitest(population)
        population = createNextGeneration(currentPopulation)
        population.extend(createRandomPopulation(point_guards, shooting_guards, 
                                               small_forwards, power_forwards, 
                                               guards, forwards, centers, util, 50 - len(population)))
        
        counter = counter+1
    print("*****************")
    return sortFitest(population)

dates = [
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
    file_name = '/Users/scotthoran/OneDrive/Documents/Daily Fantasy Football - Python Test/Genetic Algorithm Try/Data/BestLineups/{}'.format(date)
    new_file = dropLowScoringLineups(pd.concat(file, keys=key, names=['Lineup Num']))
    new_file.to_csv(file_name)

def dropLowScoringLineups(file):
    drop_level = file.groupby(['Lineup Num'])['Actual FP'].sum().mean() - file.groupby(['Lineup Num'])['Actual FP'].sum().std()
    group = file.groupby(['Lineup Num'])
    return group.filter(lambda x: x['Actual FP'].sum() > drop_level)    

def createBestLineups():
    for game_day in dates:
        file_name = '/Users/scotthoran/OneDrive/Documents/Daily Fantasy Football - Python Test/Genetic Algorithm Try/Data/HistoricalData_Merged/{}'.format(game_day[1])
        cleaned_file = reduceForProjectionAnalysis(pd.read_csv(file_name))
        best_population = geneticAlgorithmForHistorialBestLineups(cleaned_file)
        saveBestLineups_toCSV(resetIndicies(best_population), game_day[1])
        
createBestLineups()