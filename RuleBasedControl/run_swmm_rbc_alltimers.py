"""
Created by Benjamin Bowes, 03-26-2020
This script simulates stormwater system control based on OptiRTC's Continuous Monitoring and Control (CMAC) strategy.
Depth and flood values at each SWMM time step are recorded and plotted.
"""

import os
import subprocess
import json
import pandas as pd
from pyswmm import Simulation, Nodes, Links, Subcatchments
from swmm_opti_cmac import control_rules
from swmm_opti_cmac import swmm_utils

control_time_step = 900  # control time step in seconds

project_dir = "C:/PycharmProjects/swmm_opti_cmac"
inp_dir = "C:/PycharmProjects/LongTerm_SWMM/observed_data/obs_data_1month_all_controlled"
fcst_dir = "C:/PycharmProjects/LongTerm_SWMM/observed_data/obs_data_96step_fcsts"

# loop through input files
for file in os.scandir(inp_dir):
    if file.name.endswith('.inp'):
        print(file.name)
        swmm_inp = file.path  # swmm input file
        fcst_data = pd.read_csv(os.path.join(fcst_dir, file.name.split('.')[0] + ".csv"))  # forecast data frame

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
            J2 = node_object["J2"]
            Out1 = node_object["Out1"]

            St1.full_depth = 4.61
            St2.full_depth = 4.61

            link_object = Links(sim)  # init link object
            C2 = link_object["C2"]
            C3 = link_object["C3"]
            R1 = link_object["R1"]
            R2 = link_object["R2"]

            subcatchment_object = Subcatchments(sim)
            S1 = subcatchment_object["S1"]
            S2 = subcatchment_object["S2"]

            current_step = 0  # init counter for accessing forecast
            St1_drain_timer, St2_drain_timer = 0, 0  # init pre-storm drainage timers
            St1_retention_timer, St2_retention_timer = 0, 0  # init in-storm retention timers
            St1_drawdown_timer, St2_drawdown_timer = 0, 0  # init post-storm drawdown timers
            for step in sim:  # loop through all steps in the simulation
                event_dict = control_rules.read_fcst(["rain1", "rain2"], fcst_data, current_step)  # look at forecast

                if sum(event_dict['total']) > 0:  # if rain in forecast check for flooding
                    # if St1_drain_timer <= 0 or St2_drain_timer <= 0:  # check if either valve control timer expired
                    current_dt = sim.current_time

                    # save most current system states
                    init_df = pd.DataFrame([St1.depth, St2.depth, J1.depth, J2.depth, Out1.depth,
                                            C2.flow, C3.flow, R1.current_setting, R2.current_setting,
                                            current_dt]).transpose()
                    init_df.to_csv(os.path.join(project_dir, "cmac_temp/temp_inp.csv"), index=False)

                    # run sim for forecast period to get incoming volume
                    temp_file = swmm_utils.write_ctl_dates(swmm_inp, os.path.join(project_dir, "cmac_temp"), current_dt)

                    fcst_submodel = subprocess.check_output("C:/Anaconda2/envs/py36/python.exe C:/PycharmProjects/swmm_opti_cmac/run_swmm_fcst.py".split())
                    submodel_return = fcst_submodel.decode('utf-8')
                    flood_dict = json.loads(submodel_return)
                    # print(current_step, current_dt, flood_dict)

                    # check report file to see if storage units flooded and calculate new valve positions if needed
                    if flood_dict["St1"] > 0:
                        St1_drain_steps = control_rules.drain_time(flood_dict["St1"], 65000, St1.head, St1.depth)
                        if St1_drain_steps > St1_drain_timer:
                            St1_drain_timer = St1_drain_steps  # update drain timer if needed
                            St1_retention_timer = 0  # reset retention timer
                            St1_drawdown_timer = 0  # reset drawdown timer
                        # print("St1 may flood, timer updated to:", St1_drain_timer)
                        R1.target_setting = 1.  # apply new valve positions

                    if flood_dict["St2"] > 0:
                        St2_drain_steps = control_rules.drain_time(flood_dict["St2"], 50000, St2.head, St2.depth)
                        if St2_drain_steps > St2_drain_timer:
                            St2_drain_timer = St2_drain_steps
                            St2_retention_timer = 0
                            St2_drawdown_timer = 0
                        # print("St2 may flood, timer updated to:", St2_drain_timer)
                        R2.target_setting = 1.  # apply new valve positions

                # check drain timers to see if a pond is draining and decrement
                if St1_drain_timer > 0:
                    if St1_drain_timer == 1:  # pond has drawn down before storm, start retention timer
                        St1_retention_timer = 97  # retain for 1 day (96 steps + 1 to account for first decrement)
                        # TODO retention timers could be based on when storm event ends
                    St1_drain_timer -= 1
                if St2_drain_timer > 0:
                    if St2_drain_timer == 1:
                        St2_retention_timer = 97
                    St2_drain_timer -= 1

                # check retention timers
                if St1_retention_timer > 0:
                    R1.target_setting = 0  # valve closed during retention period
                    if St1_retention_timer == 1:  # pond has retained stormwater, start drawdown timer
                        St1_drawdown_timer = 97
                        R1.target_setting = control_rules.valve_position(St1.depth, 65000)
                    St1_retention_timer -= 1
                if St2_retention_timer > 0:
                    R2.target_setting = 0
                    if St2_retention_timer == 1:
                        St2_drawdown_timer = 97
                        R2.target_setting = control_rules.valve_position(St2.depth, 50000)
                    St2_retention_timer -= 1

                # check drawdown timers
                if St1_drawdown_timer > 0:
                    St1_drawdown_timer -= 1
                    if St1.depth <= 2.:  # lower target depth
                        St1_drawdown_timer = 0
                if St2_drawdown_timer > 0:
                    St2_drawdown_timer -= 1
                    if St2.depth <= 2.:
                        St2_drawdown_timer = 0

                # maintain target depth if no timers running
                if St1_drain_timer <= 0 and St1_retention_timer <= 0 and St1_drawdown_timer <= 0:
                    if St1.depth > 2.5:  # upper target depth
                        R1.target_setting = 0.5
                    if St1.depth < 2.:  # lower target depth
                        R1.target_setting = 0
                if St2_drain_timer <= 0 and St2_retention_timer <= 0 and St2_drawdown_timer <= 0:
                    if St2.depth > 2.5:
                        R2.target_setting = 0.5
                    if St2.depth < 2.:
                        R2.target_setting = 0

                # override all previous controls if ponds are flooding
                if St1.flooding > 0:
                    R1.target_setting = 1.
                if St2.flooding > 0:
                    R2.target_setting = 1.

                # print("St1 timers: ", St1_drain_timer, St1_retention_timer, St1_drawdown_timer, R1.current_setting)
                # print("St2 timers: ", St2_drain_timer, St2_retention_timer, St2_drawdown_timer, R2.current_setting)

                # print("cumulative J1 flooding: ", J1.statistics['flooding_volume'] * 7.481,
                #       "list: ", J1_fld_vol, "list sum: ", sum(J1_fld_vol),
                #       "incremental flooding: ", (J1.statistics['flooding_volume'] * 7.481 - sum(J1_fld_vol)))

                # record system state for current time step
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

                current_step += 1
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
        results_df.to_csv(os.path.join(project_dir, "cmac_results/" + file.name.split('.')[0] + ".csv"), index=False)

        # put result lists in dictionary
        result_dict = {}
        for key, value in zip(result_cols, result_list):
            result_dict[key] = value

        # plot results
        swmm_utils.plot_ctl_results(df, result_dict, file.name, os.path.join(project_dir, "cmac_results"))
