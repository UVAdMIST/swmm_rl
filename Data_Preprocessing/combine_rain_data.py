import os
import glob
import pandas as pd

base_dir = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/Raw_rain_data"
dir149 = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/Raw_rain_data/MMPS149"
dir177 = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/Raw_rain_data/MMPS177"

all_files_149 = glob.glob(os.path.join(dir149, "*.csv"))
all_files_177 = glob.glob(os.path.join(dir177, "*.csv"))

df149 = pd.concat((pd.read_csv(f, parse_dates=True, infer_datetime_format=True, index_col='Time Stamp')
                   for f in all_files_149))
df149.columns = ["MMPS149"]
df177 = pd.concat((pd.read_csv(f, parse_dates=True, infer_datetime_format=True, index_col='Time Stamp')
                   for f in all_files_177))
df177.columns = ["MMPS177"]

df = pd.concat([df149, df177], axis=1)

df.to_csv(os.path.join(base_dir, 'rain_raw.csv'))
