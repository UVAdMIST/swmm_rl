# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 07:54:22 2019

@author: cw8xk
"""
import numpy as np
import matplotlib.pyplot as plt
from pyswmm import Simulation, Nodes, Links
from rl.core import Env
from gym import spaces

swmm_inp = "C:/Users/Ben Bowes/PycharmProjects/swmm_keras_rl/case3_mpc_r1.inp"


class BasicEnv(Env):
    def __init__(self,depth=4.61):
        # initialize simulation
        self.sim = Simulation(swmm_inp)  # read input file
        self.control_time_step = 900  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step
        node_object = Nodes(self.sim)  # init node object
        self.St1 = node_object["St1"]
        self.St2 = node_object["St2"]
        self.J1 = node_object["J1"]
        self.depth = depth
        self.St1.full_depth = self.depth
        self.St2.full_depth = self.depth
        
        link_object = Links(self.sim)  # init link object
        self.R1 = link_object["R1"]
        self.R2 = link_object["R2"]
    
        self.sim.start()
        if self.sim.current_time == self.sim.start_time:
            self.R1.target_setting = 0.5
            self.R2.target_setting = 0.5
        sim_len = self.sim.end_time - self.sim.start_time
        self.T = int(sim_len.total_seconds()/self.control_time_step)
        self.t = 1
        #print('T=', self.T)        

        self.state = np.asarray([self.St1.depth, self.St2.depth, self.J1.depth,
                                 self.St1.flooding, self.St2.flooding, self.J1.flooding])
         
        self.action_space = spaces.Box(low=np.array([0,0]),high=np.array([1,1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high = 1000, shape=(len(self.state),),dtype=np.float32)
        
    def step(self,action):
        
        self.R1.target_setting = action[0]
        self.R2.target_setting = action[1]
        self.sim.__next__()
         
        self.state = np.asarray([self.St1.depth, self.St2.depth, self.J1.depth,
                                 self.St1.flooding, self.St2.flooding, self.J1.flooding])
        reward = - (self.St1.flooding + self.St2.flooding + self.J1.flooding)
        
        if self.t < self.T-1 :
            done = False
        else:
            done = True
           # self.sim.close()
        
        self.t += 1
        
        info = {}
        
        return self.state, reward, done, info       
    
    def reset(self):
        self.sim.close()
        self.sim = Simulation(swmm_inp)
        self.sim.step_advance(self.control_time_step)  # set control time step
        node_object = Nodes(self.sim)  # init node object
        self.St1 = node_object["St1"]
        self.St2 = node_object["St2"]
        self.J1 = node_object["J1"]
        link_object = Links(self.sim)  # init link object
        self.R1 = link_object["R1"]
        self.R2 = link_object["R2"]
        self.St1.full_depth = self.depth
        self.St2.full_depth = self.depth
        self.sim.start()
        self.t = 1
        if self.sim.current_time == self.sim.start_time:
            
            self.R1.target_setting = 0.5
            self.R2.target_setting = 0.5
        
        self.state = np.asarray([self.St1.depth, self.St2.depth, self.J1.depth,
                                 self.St1.flooding, self.St2.flooding, self.J1.flooding])
        return self.state
    
    def close(self):
        self.sim.report()
        self.sim.close()
  
model0 = BasicEnv()
#
d = False
#st1_depth = [model0.state[0]]
#st2_depth = [model0.state[1]]
#J3_depth = [model0.state[2]]
while not d:
    action = [0,0]
    state, reward, d, _ = model0.step(action)
#    st1_depth.append(state[0])
#    st2_depth.append(state[1])
#    J3_depth.append(state[2])
#    #print(state[3:])
model0.close()
##plt.plot(st2_depth)
#plt.plot(J3_depth)
