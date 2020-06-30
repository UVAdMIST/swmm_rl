"""
Created by Benjamin Bowes, 11-19-19
Updated 4-5-20
This script records depth and flood values at each swmm model time step and plots them.
"""

import os
import pandas as pd
from pyswmm import Simulation, Nodes, Links, Subcatchments
from swmm_opti_cmac import swmm_utils

project_dir = "C:/PycharmProjects/LongTerm_SWMM/observed_data/1month_baseline_results"
inp_dir = "C:/PycharmProjects/LongTerm_SWMM/observed_data/obs_data_1month_all_uncontrolled"
# inp_dir = "C:/Users/Ben Bowes/PycharmProjects/LongTerm_SWMM/observed_data/082019_weekly"

# loop through testing envs
control_time_step = 900  # control time step in seconds
for file in os.scandir(inp_dir):
    if file.name.endswith('.inp'):
        swmm_inp = file.path  # swmm input file

        St1_depth, St2_depth, J1_depth = [], [], []
        St1_flooding, St2_flooding, J1_flooding = [], [], []  # flood volume in cubic feet
        St1_full, St2_full, J1_full = [], [], []
        R1_act, R2_act, total_flood = [], [], []
        St1_fld_vol, St2_fld_vol, J1_fld_vol = [], [], []

        with Simulation(swmm_inp) as sim:  # set up simulation
            sim.step_advance(control_time_step)
            node_object = Nodes(sim)  # init node object
            St1 = node_object["St1"]
            St2 = node_object["St2"]
            J1 = node_object["J1"]

            St1.full_depth = 4.61
            St2.full_depth = 4.61

            link_object = Links(sim)  # init link object
            R1 = link_object["R1"]
            R2 = link_object["R2"]

            subcatchment_object = Subcatchments(sim)
            S1 = subcatchment_object["S1"]
            S2 = subcatchment_object["S2"]

            for step in sim:  # loop through all steps in the simulation
                if sim.current_time == sim.start_time:
                    R1.target_setting = 1
                    R2.target_setting = 1

                St1_depth.append(St1.depth)
                St2_depth.append(St2.depth)
                J1_depth.append(J1.depth)
                St1_flooding.append(St1.flooding)
                St2_flooding.append(St2.flooding)
                J1_flooding.append(J1.flooding)
                St1_full.append(St1.full_depth)
                St2_full.append(St2.full_depth)
                J1_full.append(J1.full_depth)
                R1_act.append(R1.current_setting)
                R2_act.append(R2.current_setting)
                St1_fld_vol.append((St1.statistics['flooding_volume'] * 7.481 - sum(St1_fld_vol)))  # incremental vol
                St2_fld_vol.append((St2.statistics['flooding_volume'] * 7.481 - sum(St2_fld_vol)))
                J1_fld_vol.append((J1.statistics['flooding_volume'] * 7.481 - sum(J1_fld_vol)))
                total_flood.append((St1.statistics['flooding_volume'] + St2.statistics['flooding_volume'] +
                                    J1.statistics['flooding_volume']) * 7.481 / 1e6)  # cumulative vol in 10^6 gallons
            sim.close()

        # read rain and tide data from inp file
        df = swmm_utils.get_env_data(file.path)

        # save results data
        result_list = [St1_depth, St2_depth, J1_depth, St1_flooding, St1_fld_vol, St2_flooding, St2_fld_vol,
                       J1_flooding, J1_fld_vol, total_flood, St1_full, St2_full, J1_full, R1_act, R2_act]
        result_cols = ["St1_depth", "St2_depth", "J1_depth", "St1_flooding", "St1_fld_vol", "St2_flooding",
                       "St2_fld_vol", "J1_flooding", "J1_fld_vol", "total_flood", "St1_full", "St2_full",
                       "J1_full", "R1_act", "R2_act"]
        results_df = pd.DataFrame(result_list).transpose()
        results_df.columns = result_cols
        results_df = pd.concat([results_df, df], axis=1)
        results_df.to_csv(os.path.join(project_dir, file.name.split('.')[0] + ".csv"), index=False)

        # put result lists in dictionary
        result_dict = {}
        for key, value in zip(result_cols, result_list):
            result_dict[key] = value

        # plot results
        swmm_utils.plot_ctl_results(df, result_dict, file.name, project_dir)
