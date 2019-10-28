"""
Benjamin Bowes, 23 Aug., 2019

This script creates tidal time series of different return periods by shifting an average tide cycle.
Tide data is from the NOAA Sewells Point Station. Exceedance probabilities are from:
https://tidesandcurrents.noaa.gov/est/stickdiagram.shtml?stnid=8638610
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read base tide
base_tide = pd.read_csv("LongTerm_SWMM/base_tide.csv")
base_tide['Datetime'] = pd.to_datetime(base_tide['Datetime'], infer_datetime_format=True)
base_tide.set_index('Datetime', drop=True, inplace=True)

# Set up exceedance probability levels (in feet)
exceed_prob = np.array([[7.71, 5.74, 4.49, 3.51], [-3.33, -3.12, -2.56, -1.94]]).transpose()
exceed_names = ['100yr', '10yr', '2yr', '1yr']

# Get max/min tide value and shift series based on difference between exceedance probability level
tide_max = base_tide.max()[0]
tide_min = base_tide.min()[0]

for val, name in zip(exceed_prob, exceed_names):
    diff_max = val[0] - tide_max
    diff_min = val[1] - tide_min

    df_max = pd.DataFrame(base_tide['Tide'] + diff_max)
    df_max.columns = ['Max']
    df_min = pd.DataFrame(base_tide['Tide'] + diff_min)
    df_min.columns = ['Min']
    df = pd.concat([base_tide, df_max, df_min], axis=1)
    df.reset_index(inplace=True)

    # save as csv files
    max_csv = df[['Datetime', 'Max']]
    min_csv = df[['Datetime', 'Min']]

    max_csv.to_csv("LongTerm_SWMM/tide_csv_files/tide_" + name + "_max.csv", index=False)
    min_csv.to_csv("LongTerm_SWMM/tide_csv_files/tide_" + name + "_min.csv", index=False)

    # plot distribution
    plt.plot(df['Tide'], label='Baseline')
    plt.plot(df['Max'], label='Max')
    plt.plot(df['Min'], label='Min')
    plt.xlim(0, 24)
    plt.xlabel('hour')
    plt.ylabel('tide level above NAVD88 (ft)')
    plt.title(name)
    plt.legend()
    plt.savefig("LongTerm_SWMM/tide_plots/" + name + ".png")
    plt.close()

    df['year'] = df['Datetime'].dt.year
    df['month'] = df['Datetime'].dt.month
    df['day'] = df['Datetime'].dt.day
    df['hour'] = df['Datetime'].dt.hour
    df['minutes'] = df['Datetime'].dt.minute
    df['station'] = "Tide1"

    # Convert to swmm format
    swmm_max = df[['station', 'year', 'month', 'day', 'hour', 'minutes', 'Max']]
    swmm_min = df[['station', 'year', 'month', 'day', 'hour', 'minutes', 'Min']]

    swmm_max.to_csv("LongTerm_SWMM/tide_dat_files/tide_" + name + "_max.dat", sep=' ', index=False, header=False)
    swmm_min.to_csv("LongTerm_SWMM/tide_dat_files/tide_" + name + "_min.dat", sep=' ', index=False, header=False)

# convert base tide to dat file
base_tide.reset_index(inplace=True)

base_tide['year'] = base_tide['Datetime'].dt.year
base_tide['month'] = base_tide['Datetime'].dt.month
base_tide['day'] = base_tide['Datetime'].dt.day
base_tide['hour'] = base_tide['Datetime'].dt.hour
base_tide['minutes'] = base_tide['Datetime'].dt.minute
base_tide['station'] = "Tide1"

swmm_baseline = base_tide[['station', 'year', 'month', 'day', 'hour', 'minutes', 'Tide']]
swmm_baseline.to_csv("LongTerm_SWMM/tide_dat_files/tide_base_line.dat", sep=' ', index=False, header=False)
