#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:23:31 2022

@author: scotthoran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import floor

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
#     ['2_13', '2020-02-13'],
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
#     ['1_9', '2020-01-09'],
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
#     ['11_12', '2019-11-12'],
    ['11_11', '2019-11-11'],
#     ['11_10', '2019-11-10'],
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
#     ['10_25', '2019-10-25'],
#     ['10_24', '2019-10-24'],
#     ['10_23', '2019-10-23'],
#     ['10_22', '2019-10-22'],   
]

files= [
    'RandomlyCreatedLineups',
    'BestCreatedLineups_AvgProj_Baseline',
    'BestCreatedLineups_ElasticNet',
    'BestCreatedLineups_LinearRegression',
    'BestCreatedLineups_LinearRegression_400',
    'BestCreatedLineups_LinearRegression_800',
    'BestCreatedLineups'
]


def openMoneyLines():
    file_name = 'Metrics/MoneyLines.csv'
    file = pd.read_csv(file_name)
    return file

def openCreatedLineupsForDate(game_day, file_name):
    file_name = '{}/{}'.format(file_name, game_day)
    file = pd.read_csv(file_name,index_col=[0,1], skipinitialspace=True)
    return file

def constructDate(date):
    year = int(date[2:4])
    month = int(date[5:7])
    day = int(date[8:])
    return str(month)+'/'+str(day)+'/'+str(year)

def findLinesForDate(date, lines):
    gpp_line = lines[lines['Date'] == constructDate(date)]['GPP Line'].values[0]
    cash_line = lines[lines['Date'] == constructDate(date)]['Cash Line '].values[0]
    return gpp_line, cash_line

def calculateWL( selected_entries, 
                gpp_line, 
                cash_line, 
                cash_line_wins, 
                gpp_line_wins,
                cash_line_losses, 
                gpp_line_losses,
                top_three,
                tenth_place,
                fiftyth_place):
    for score in list(selected_entries.groupby(['Lineup Num'])['Actual FP'].sum()):
        if(score >= cash_line): 
            cash_line_wins = cash_line_wins + 1
        if(score < cash_line):
            cash_line_losses = cash_line_losses + 1
        if(score >= gpp_line): 
            gpp_line_wins = gpp_line_wins + 1
        if(score < gpp_line):
            gpp_line_losses = gpp_line_losses + 1
        if(score >= gpp_line+85): 
            top_three = top_three + 1
        if(score >= gpp_line+70): 
            tenth_place = tenth_place + 1
        if(score >= gpp_line+58): 
            fiftyth_place = fiftyth_place + 1
        
    return cash_line_wins, gpp_line_wins,cash_line_losses, gpp_line_losses, top_three, tenth_place, fiftyth_place

def calculateWinPercentage(cash_line_wins, gpp_line_wins,cash_line_losses, gpp_line_losses):
    cash_line_win_percent = (cash_line_wins / (cash_line_wins + cash_line_losses))*100
    gpp_line_win_percent = (gpp_line_wins / (gpp_line_wins + gpp_line_losses))*100
    total_win_percent = ((cash_line_wins + gpp_line_wins) / (cash_line_wins + cash_line_losses + gpp_line_wins + gpp_line_losses))*100
    return cash_line_win_percent, gpp_line_win_percent, total_win_percent
    
def calculateProfitability():
    lines = openMoneyLines()
    full_report = []
    for file_name in files:
        for selection in range(0,11):
            cash_line_wins = 0
            gpp_line_wins = 0
            cash_line_losses = 0
            gpp_line_losses = 0
            top_three = 0
            tenth_place=0
            fiftyth_place=0
            for game_day in dates_all:
                gpp_line, cash_line = findLinesForDate(game_day[1], lines)
                created_lineups = openCreatedLineupsForDate(game_day[1], file_name)
                selected_entries = created_lineups.loc[:selection]
                cash_line_wins, gpp_line_wins,cash_line_losses, gpp_line_losses,top_three, tenth_place, fiftyth_place = calculateWL(selected_entries, 
                                                                                              gpp_line, 
                                                                                              cash_line, 
                                                                                              cash_line_wins, 
                                                                                              gpp_line_wins,
                                                                                              cash_line_losses, 
                                                                                              gpp_line_losses,
                                                                                              top_three,
                                                                                              tenth_place,
                                                                                              fiftyth_place)
            cash_line_win_percent, gpp_line_win_percent, total_win_percent = calculateWinPercentage(cash_line_wins, 
                                                                                                    gpp_line_wins,
                                                                                                    cash_line_losses, 
                                                                                                    gpp_line_losses)
                                                                                                
            full_report.append({'total_win_percent': total_win_percent,
                                'cash_win_percent': cash_line_win_percent,
                                'gpp_win_percent': gpp_line_win_percent,
                                'top_three': top_three,
                                'tenth_place:': tenth_place,
                                'fifthy_place:': fiftyth_place, 
                                'file_name': file_name, 
                                'selection': selection+1})
    return full_report

x = calculateProfitability()

x