"""
Created by Benjamin Bowes, 4-19-19
This script records depth and flood values at each swmm model time step and plots them.
"""

from pyswmm import Simulation, Nodes, Links, Subcatchments
import matplotlib.pyplot as plt
from smart_stormwater_rl.pyswmm_utils import save_out

control_time_step = 900  # control time step in seconds
swmm_inp = "C:/Users/Ben Bowes/PycharmProjects/swmm_keras_rl/case3_mpc_r1.inp"  # swmm input file

St1_depth = []
St2_depth = []
J1_depth = []
St1_flooding = []  # flood volume in cubic feet
St2_flooding = []
J1_flooding = []
St1_full = []
St2_full = []
J1_full = []

with Simulation(swmm_inp) as sim:  # loop through all steps in the simulation
    sim.step_advance(control_time_step)
    node_object = Nodes(sim)  # init node object
    St1 = node_object["St1"]
    St2 = node_object["St2"]
    J1 = node_object["J1"]

    # St1.full_depth = 4
    # St2.full_depth = 4

    link_object = Links(sim)  # init link object
    R1 = link_object["R1"]
    R2 = link_object["R2"]

    subcatchment_object = Subcatchments(sim)
    S1 = subcatchment_object["S1"]
    S2 = subcatchment_object["S2"]

    for step in sim:
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
        # print("current time: ", sim.current_time, "flooding: ", J3.flooding, "flood_vol: ",
        #       J3.statistics['flooding_volume'], "flood_rate: ", J3.statistics['peak_flooding_rate'])

out_lists = [St1_depth, St2_depth, J1_depth, St1_flooding, St2_flooding, J1_flooding]

# save_out(out_lists, "Uncontrolled")

# plot results
plt.subplot(3, 1, 1)
plt.plot(St1_depth, label='St1')
plt.plot(St2_depth, label='St2')
plt.plot(St1_full, linestyle=':', color='k', label='max depth')
plt.title('Storage Unit Depth')
plt.ylabel("ft")
plt.ylim(0, 6)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(J1_depth, label='J1', color='green')
plt.plot(J1_full, linestyle=':', color='k', label='max depth')
plt.title('Downstream Node Depth')
plt.ylabel("ft")
plt.ylim(0, 3)
plt.legend()

# bar graph for total flooding
# plt.subplot(2, 2, 4)
# plt.bar([0, 1, 2], [sum(St1_flooding) * 7.481 / 10 ** 6, sum(St2_flooding) * 7.481 / 10 ** 6,
#                     sum(J3_flooding) * 7.481 / 10 ** 6], tick_label=["ST1", "St2", "J1"])
# plt.title('total_flooding')
# plt.ylabel("10^6 gallons")

plt.subplot(3, 1, 3)
plt.plot(St1_flooding, label='St1')
plt.plot(St2_flooding, label='St2')
plt.plot(J1_flooding, label='J1')
plt.title('Flooding')
plt.ylabel("cfs")
plt.xlabel("time step")
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig("C:/Users/Ben Bowes/PycharmProjects/swmm_keras_rl/baseline_model_case3.png", dpi=300)
plt.close()
