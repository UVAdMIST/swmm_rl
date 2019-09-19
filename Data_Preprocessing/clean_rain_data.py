import pandas as pd
import datetime
import matplotlib.pyplot as plt

path_raw = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/Raw_rain_data/rain_raw.csv"

raw_df = pd.read_csv(path_raw, parse_dates=True, infer_datetime_format=True, index_col='Time Stamp')
print(raw_df.sum())
# print(raw_df.mean(), raw_df.min(), raw_df.max())

# remove large values
raw_df.loc[raw_df['MMPS149'] > 10., 'MMPS149'] = 0
raw_df.loc[raw_df['MMPS177'] > 10., 'MMPS177'] = 0
print(raw_df.sum())

# take mean
raw_df['mean'] = raw_df.mean(axis=1)
df = pd.DataFrame(raw_df['mean'])

# plot
# raw_df['mean'].plot()
# plt.ylabel('Rainfall (in)')
# plt.xlabel('Date')
# plt.show()

# remove rows of 0s
df2 = df.loc[df['mean'] != 0]

# save basic csv
df2.to_csv("C:/Users/Ben Bowes/Documents/LongTerm_SWMM/rain_cleaned.csv")

# convert to swmm format
"""
Based on SWMM Manual: a standard user-prepared format where each line of the file contains
the station ID, year, month, day, hour, minute, and non-zero precipitation reading, all separated
by one or more spaces.

SWMM Example:

STA01 2004 6 12 00 00 0.12
STA01 2004 6 12 01 00 0.04
STA01 2004 6 22 16 00 0.07
"""

df2.reset_index(inplace=True)

df2['year'] = df2['Time Stamp'].dt.year
df2['month'] = df2['Time Stamp'].dt.month
df2['day'] = df2['Time Stamp'].dt.day
df2['hour'] = df2['Time Stamp'].dt.hour
df2['minutes'] = df2['Time Stamp'].dt.minute

df2['station'] = "Gage1"

swmm_df = df2[['station', 'year', 'month', 'day', 'hour', 'minutes', 'mean']]

swmm_df.to_csv("C:/Users/Ben Bowes/Documents/LongTerm_SWMM/rain_swmm_frmt.dat", sep=' ', index=False, header=False)
