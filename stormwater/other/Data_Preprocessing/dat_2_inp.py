"""
Benjamin Bowes, 27 Aug., 2019

This script reads .dat files of synthetic rainfall and tide and writes it into a .inp file as a time series.
"""

import os
import pandas as pd

inp_file = "LongTerm_SWMM/case3_syndata_template.inp"

# find time series section
with open(inp_file, 'r') as tmp_file:
    lines = tmp_file.readlines()
    for i, l in enumerate(lines):
        if l.startswith("[TIMESERIES]"):
            # print(i, l)
            start = i + 3

rain_dir = "C:/Users/Ben Bowes/PycharmProjects/LongTerm_SWMM/Atlas14_dat_files"
tide_dir = "C:/Users/Ben Bowes/PycharmProjects/LongTerm_SWMM/tide_dat_files"

# loop through all combinations of rainfall and tide .dat files
for rain_file in os.scandir(rain_dir):
    print(rain_file.name)
    rain_name = rain_file.name.split('_')[0]
    rain_num = rain_file.name.split('_')[1].split('.')[0]
    rain_df = pd.read_csv(rain_file.path, sep=" ", header=None)
    rain_df.columns = ['station', 'year', 'month', 'day', 'hour', 'minute', 'rain']
    # format date and time
    rain_df['hour'] = rain_df['hour'].astype(str).str.zfill(2)
    rain_df['minute'] = rain_df['minute'].astype(str).str.zfill(2)
    rain_df['Date'] = pd.to_datetime(rain_df[['year', 'month', 'day']], format='%m%d%Y').dt.strftime('%m/%d/%Y')
    # rain_df['Date'] = pd.to_datetime(rain_df['Date'])
    # rain_df['Time'] = rain_df['Date'].dt.hour.astype(str)
    rain_df['Time'] = rain_df['hour'].astype(str) + ":" + rain_df['minute'].astype(str)

    for tide_file in os.scandir(tide_dir):
        tide_name = tide_file.name.split('_')[1] + tide_file.name.split('_')[2].split('.')[0]
        tide_df = pd.read_csv(tide_file.path, sep=" ", header=None)
        tide_df.columns = ['station', 'year', 'month', 'day', 'hour', 'minute', 'tide']
        # format date and time
        tide_df['hour'] = tide_df['hour'].astype(str).str.zfill(2)
        tide_df['minute'] = tide_df['minute'].astype(str).str.zfill(2)
        tide_df['Date'] = pd.to_datetime(tide_df[['year', 'month', 'day']], format='%m%d%Y').dt.strftime('%m/%d/%Y')
        tide_df['Time'] = tide_df['hour'].astype(str) + ":" + tide_df['minute'].astype(str)

        # create inp file name, Ex: 2yr24hr_10yrmin_0.inp (rain_tide_rain file number.inp)
        inp_name = rain_name + "_" + tide_name + "_" + rain_num + ".inp"

        # write data to inp file
        with open("LongTerm_SWMM/syn_inp_files/" + inp_name, 'w') as inpfile:
            new_lines = lines.copy()  # copy inp template
            for indx, row in rain_df.iterrows():  # write rain data
                new_line = str(row.station) + " " + str(row.Date) + " " + str(row.Time) + " " + str(row.rain)
                # print(new_line)
                if indx == rain_df.shape[0]-1:
                    new_lines.insert(start + indx, new_line + '\n;')
                else:
                    new_lines.insert(start + indx, new_line + '\n')

            for indx, row in tide_df.iterrows():  # write tide data
                new_line = str(row.station) + " " + str(row.Date) + " " + str(row.Time) + " " + str(row.tide)
                # print(new_line)
                if indx == tide_df.shape[0] - 1:
                    new_lines.insert(start + rain_df.shape[0] + indx + 1, new_line + '\n\n')
                else:
                    new_lines.insert(start + rain_df.shape[0] + indx + 1, new_line + '\n')
            inpfile.writelines(new_lines)
