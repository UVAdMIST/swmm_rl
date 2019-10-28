"""
Benjamin Bowes, 10 Sept., 2019

This script changes .inp files to use a tide gate at the outfall.
"""

import os
import fileinput
import sys


def replaceAll(file, searchExp, replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)


base_dir = "C:/Users/Ben Bowes/PycharmProjects/LongTerm_SWMM"
folder_list = ['syn_inp_train', 'syn_inp_test']
# folder_list = ['syn_inp_small_test']
template = "LongTerm_SWMM/case3_syndata_template.inp"

# find time series section
with open(template, 'r') as tmp_file:
    lines = tmp_file.readlines()
    for i, l in enumerate(lines):
        if l.startswith("[OUTFALLS]"):
            # print(i, l)
            start = i + 3

# loop through all folders
for folder in folder_list:
    for file in os.scandir(os.path.join(base_dir, folder)):
        if file.name.endswith('.inp'):
            print(file.name)

            old_line = "Out1             0          TIMESERIES Tide1            YES"
            new_line = "Out1             0          TIMESERIES Tide1            NO"

            replaceAll(file.path, old_line, new_line)
