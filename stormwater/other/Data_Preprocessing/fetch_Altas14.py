"""
Benjamin Bowes, 21 Aug., 2019

This script fetches point precipitation frequency estimates from the NOAA Atlas 14
Precipitation Frequency Data Server for any latitude and longitude. The estimates are
transformed into a rainfall time series using the SCS Type II distribution as per
https://www.wcc.nrcs.usda.gov/ftpref/wntsc/H&H/NEHhydrology/ch4_Sept2015draft.pdf
"""

import random
import requests
from bs4 import BeautifulSoup as bs
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gen_samples(lower, upper, n=50):
    # generate n samples from distribution between [lower:upper]
    samples = []
    for i in range(n):
        sample = round(random.uniform(lower, upper), 3)
        samples.append(sample)

    return samples


# Specify latitude and longitude for location
lat = 36.8661
long = -76.289

# get data from NOAA Atlas 14
url = "https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/cgi_readH5.py?lat={0}&lon={1}&type=pf&data=depth&" \
      "units=english&series=pds".format(lat, long)

response = requests.get(url)
soup = bs(response.text, 'html.parser').get_text()

quant = soup.find('quantiles')
up = soup.find('upper')
low = soup.find('lower')
file = soup.find('file')

quantiles = re.split('[ |] |,', soup[quant:up])
upper = re.split('[ |] |,', soup[up:low])
lower = re.split('[ |] |,', soup[low:file])

str_quant = []
str_upper = []
str_lower = []
for i in range(0, len(quantiles)):
    # print(quantiles[i].split("'")[1])
    str_quant.append(quantiles[i].split("'")[1])
    str_upper.append(upper[i].split("'")[1])
    str_lower.append(lower[i].split("'")[1])

# create array of rainfall values, column is return period, row is duration
quant_arr = np.array(str_quant, dtype='float32').reshape((19, 10))
upper_arr = np.array(str_upper, dtype='float32').reshape((19, 10))
lower_arr = np.array(str_lower, dtype='float32').reshape((19, 10))

# convert each duration/return period combination into a time series
ratios = pd.read_csv("LongTerm_SWMM/SCSratios_noblanks.csv", index_col='time')  # ratios to 24hr storm

# loop through return periods and durations
return_period = ['1yr', '2yr', '5yr', '10yr', '25yr', '50yr', '100yr', '200yr', '500yr', '1000yr']
duration = ['5-min', '10-min', '15-min', '30-min', '60-min', '2-hour', '3-hour', '6-hour', '12-hour', '24-hour']

for return_indx, return_name in enumerate(return_period):
    print("creating files for ", return_name, " storms")
    for duration_indx, duration_name in enumerate(duration):
        key = str(return_name + duration_name)
        # print(key)
        mean_value = quant_arr[return_indx][duration_indx]
        upper_value = upper_arr[return_indx][duration_indx]
        lower_value = lower_arr[return_indx][duration_indx]
        df = pd.DataFrame(ratios[['Datetime', 'hour', 'min']])
        df['mean'] = ratios[duration_name] * mean_value
        df['upper'] = ratios[duration_name] * upper_value
        df['lower'] = ratios[duration_name] * lower_value

        # plot distribution
        plt.plot(df[['mean', 'upper', 'lower']])
        plt.xlim(0, 24)
        plt.xlabel('hour')
        plt.ylabel('cumulative rainfall (in)')
        plt.title(key)
        plt.savefig("LongTerm_SWMM/Atlas14_plots/" + key + ".png")
        plt.close()

        # generate time series and save as dat file
        samples = gen_samples(lower_value, upper_value)
        for sample_num, sample in enumerate(samples):
            sample_df = df[['Datetime', 'hour', 'min']]
            sample_df[key] = ratios[duration_name] * sample

            # save as csv file
            sample_df.to_csv("LongTerm_SWMM/Atlas14_csv_files/" + key + "_" + str(sample_num) + ".csv", index=False)

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

            sample_df['station'] = 'Atlas14'
            sample_df['year'] = '2000'
            sample_df['month'] = '1'
            sample_df['day'] = '1'

            swmm_df = sample_df[['station', 'year', 'month', 'day', 'hour', 'min', key]]
            swmm_df.dropna(inplace=True)

            swmm_df.to_csv("LongTerm_SWMM/Atlas14_dat_files/" + key + "_" + str(sample_num) + ".dat", sep=' ',
                           index=False, header=False)
