"""
Benjamin Bowes, 05 Sept., 2019

This script reads .csv files of synthetic rainfall and tide and creates a shifted time series of forecasts.
"""

import os
import pandas as pd


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, col_name='var', dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(col_name + '(t-%d)' % i)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(col_name + '(t)')]
        else:
            names += [(col_name + '(t+%d)' % i)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


rain_dir = "C:/Users/Ben Bowes/PycharmProjects/LongTerm_SWMM/Atlas14_csv_files"
tide_dir = "C:/Users/Ben Bowes/PycharmProjects/LongTerm_SWMM/tide_csv_files"

for rain_file in os.scandir(rain_dir):
    rain_name = rain_file.name.split('_')[0]
    rain_num = rain_file.name.split('_')[1].split('.')[0]
    print(rain_name)

    rain_df = pd.read_csv(rain_file.path, usecols=['Datetime', rain_name])
    rain_df[rain_name].fillna(method='bfill', inplace=True)
    rain_df[rain_name].fillna(method='ffill', inplace=True)
    rain_df['inc_rain'] = rain_df[rain_name]
    for i, row in rain_df.iterrows():  # get incremental rainfall
        if i == 0:
            rain_df['inc_rain'][i] = 0.
        else:
            rain_df['inc_rain'][i] = rain_df[rain_name][i] - rain_df[rain_name][i - 1]

    rain_df['Datetime'] = pd.to_datetime(rain_df['Datetime'], infer_datetime_format=True)
    rain_df.set_index(pd.DatetimeIndex(rain_df['Datetime']), inplace=True)
    rain_df = rain_df.resample('15T').sum()
    rain_df.drop(rain_name, axis=1, inplace=True)

    # create shifted time series
    rain_fcst = series_to_supervised(rain_df, 0, 19, col_name='rain', dropnan=False).fillna(0)
    rain_fcst.reset_index(inplace=True, drop=True)

    for tide_file in os.scandir(tide_dir):
        tide_name = tide_file.name.split('_')[1] + tide_file.name.split('_')[2].split('.')[0]

        tide_df = pd.read_csv(tide_file.path, index_col='Datetime', parse_dates=True, infer_datetime_format=True)
        tide_df = tide_df.resample('15T').interpolate(method='linear')
        tide_df.reset_index(inplace=True, drop=True)
        tide_df = pd.concat([tide_df, tide_df])
        tide_df.reset_index(inplace=True, drop=True)

        # create shifted time series
        tide_fcst = series_to_supervised(tide_df, 0, 19, col_name='tide', dropnan=False)

        # combine rain and tide forecasts and save
        fcst_df = pd.concat([rain_fcst, tide_fcst[:len(rain_fcst)]], axis=1)

        # create inp file name, Ex: 2yr24hr_10yrmin_0 (rain_tide_rain file number)
        fcst_name = rain_name + "_" + tide_name + "_" + rain_num + ".csv"

        fcst_df.to_csv("LongTerm_SWMM/fcst_files/" + fcst_name, index=False)
