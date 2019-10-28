import numpy as np
import random
from reward_functions import reward_function3 as reward_function
from pyswmm import Simulation, Nodes, Links
# import matplotlib.pyplot as plt
import gym


class StormwaterEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, R=6, J=4, J_offset=1, swmm_inp="data/simple3.inp"):


        super(StormwaterEnv, self).__init__()
        self.total_rewards = []
        self.eps_reward = 0

        self.global_current_step = 0
        self.current_step = 0
        self.control_time_step = 900

        self.swmm_inp = swmm_inp        
        self.temp_height = np.zeros(2, dtype='int32')  # St1.depth, St2.depth
        self.temp_valve = np.zeros(2, dtype='int32')  # R1.current_setting, R2.current_setting
       
        self.R = R
        self.J = J
        self.J_offset = J_offset
        
        self.log_dumps = [[]]

    def reset(self):
        # make this dynamic later
        # self.swmm_inp="data/simple3.inp"
        self.sim = Simulation(self.swmm_inp)
        self.sim.step_advance(self.control_time_step)  # set control time step
        
        self.node_object = Nodes(self.sim)
        self.link_object = Links(self.sim)

        
        self.total_rewards.append(self.eps_reward)
        self.eps_reward = 0

        self.global_current_step += 1
        self.current_step = 0


        self.done = False


        self.sim.start()


        self.sim_len = self.sim.end_time - self.sim.start_time
        self.T = self.sim_len.total_seconds()//self.control_time_step
        # self.T  = 100

        
        for i in range(1, self.R+1):
            self.link_object['R'+str(i)].target_setting = random.randrange(1)

        self.settings = [self.link_object['R' + str(i)].current_setting for i in range(1, self.R+1)]

        # one for loop would be faster but less readable
        # making new lists all the time is probably bad
        self.depths = [self.node_object['St' + str(i)].depth for i in range(1, self.R+1)]
        self.depths.extend([self.node_object['J'+str(i)].depth for i in range(self.J_offset+1, self.J+1)])
       
        self.flooding = [self.node_object['St' + str(i)].flooding for i in range(1, self.R+1)]
        self.flooding.extend([self.node_object['J'+str(i)].flooding for i in range(self.J_offset+1, self.J+1)])
     
        self.reward = reward_function(self.depths, self.flooding)
        
        self.eps_reward += self.reward
            
        return np.asarray(self.settings + self.depths + self.flooding)


        # Take action
            # + Initiate the step in the simulation
        # Make observation / calculate state
        # calculate the reward
        # check if done
        # Do some logging for graphing

    def step(self, action):
        self.current_step += 1 # Decide whether this should be before or after taking actions

        ##### Take Action ##### 
        for i in range(1, self.R+1):
            self.link_object['R'+str(i)].target_setting = action[i-1]

        self.sim.__next__() 
        

        ##### Make Observation #####
        self.settings = [self.link_object['R' + str(i)].current_setting for i in range(1, self.R+1)]

        self.depths = [self.node_object['St' + str(i)].depth for i in range(1, self.R+1)]
        self.depths.extend([self.node_object['J'+str(i)].depth for i in range(self.J_offset+1, self.J+1)])
       
        self.flooding = [self.node_object['St' + str(i)].flooding for i in range(1, self.R+1)]
        self.flooding.extend([self.node_object['J'+str(i)].flooding for i in range(self.J_offset+1, self.J+1)])
       
        # one for loop would be faster but less readable  (For the above)
        # making new lists all the time is probably bad   
        #   Actually timed this and performed better than changing the elements
        

        ##### Calculate Reward #####
                        # - max flooding looked better than this.
        self.reward = - np.max(self.flooding) * np.sum(self.flooding)
        self.eps_reward += self.reward

        ##### Check if Done #####
        self.done = False if self.current_step < self.T-1 else True


        ##### for graphing #####
        if self.current_step == 1:
            # self.log_dumps.append([])
            self.log_dumps[-1] = []
        self.log_dumps[-1].append((self.reward, [self.settings, self.depths, self.flooding], self.current_step))
        return np.asarray(self.settings + self.depths + self.flooding), self.reward, self.done, {} # {} is debugging information
    

    def close(self):
        self.sim.report()
        self.sim.close()


    def render(self, mode="human"):
        # This's probably not useful at all at the moment but should be here
        return "Settings: " + str(self.settings) + "\n" + \
            "Depths: " + str(self.depths) + "\n" + \
            "Flooding: " + str(self.flooding) + "\n"


    # def graph(self, location):
    #     for plot, dump in enumerate(self.log_dumps[-1:]):
    #         settings, depths, floodings = [[] for i in range(self.R)], [[] for i in range(self.R + self.J - 1)], [[] for i in range(self.R + self.J - 1)]
    #         rewards = []
        
    #         for reward, state, timestep in dump: # timestep should stay local here

    #             rewards.append(reward)
    #             setting, depth, flooding = state
                
    #             for i, R in enumerate(settings):
    #                 R.append(setting[i])
    #             for i, S_depth in enumerate(depths):
    #                 S_depth.append(depth[i])
    #             for i, S_flooding in enumerate(floodings):
    #                 S_flooding.append(flooding[i])
            

    #         for i in range(1, self.R+1):
    #             plt.subplot(2, 3, i)
    #             plt.plot(settings[i-1])
    #             plt.ylim(0, 1)
    #             plt.title('R' + str(i))
    #             plt.ylabel("Valve Opening")
    #             plt.xlabel("time step")
            
    #         plt.tight_layout()
    #         if(plot == 5):
    #             plt.savefig(location + "TEST_" + str(timestep) + "_STATES", dpi=300)
    #         else:
    #             plt.savefig(location + str(timestep) + "_STATES", dpi=300)
           
    #         plt.clf()
            
    #         for i in range(1, self.R+1):
    #             plt.subplot(2, 3, i)
    #             plt.plot(depths[i-1])
    #             plt.ylim(0, 5)
    #             plt.title('St' + str(i) + " Depth")
    #             plt.ylabel("Feet")
    #             plt.xlabel("time step")


    #         plt.tight_layout()
    #         if(plot == 5):
    #             plt.savefig(location + "TEST_" + str(timestep) + "_DEPTHS", dpi=300)
    #         else:
    #             plt.savefig(location + str(timestep) + "_DEPTHS", dpi=300)
           
    #         plt.clf()

    #         for i in range(self.J_offset+1, self.J+1):
    #             plt.subplot(2, 2, max(i-1, 1))
    #             plt.plot(floodings[self.R + i-2])
    #             plt.title('J' + str(i) + " flooding")
    #             plt.ylabel("Feet")
    #             plt.xlabel("time step")
            

    #         plt.subplot(2, 2, 4)
    #         plt.bar([i for i in range(1, self.R+self.J-self.J_offset)], [sum(floodings[i]) for i in range(len(floodings))], 
    #                         tick_label= ["St1","St2", "J1", "J2"])
                            
    #         #["St" + str(i) for i in range(1, self.R+1)] + ["J" + str(i) for i in range(self.J_offset+1, self.J+1)]) 
    #         # ["St1","St2","St3","St4","St5","St6", "J2", "J3", "J4"])
    #         plt.ylim(0)
    #         plt.title('total_flooding')
    #         plt.ylabel("10^3 cubic feet")
    #         plt.xlabel("time step")

    #         plt.tight_layout()

    #         if(plot == 5):
    #             plt.savefig(location + "TEST_" + str(timestep) + "_FLOODING", dpi=300)
    #         else:
    #             plt.savefig(location + str(timestep) + "_FLOODING", dpi=300)
            
    #         plt.clf() 

    #         print("Total Flooding:", sum([sum(floodings[i]) for i in range(len(floodings))]))

    #         plt.subplot(2, 1, 1)
    #         plt.plot(rewards)
    #         # plt.ylim(0, 5)
    #         plt.title('Rewards')
    #         plt.ylabel("Reward")
    #         plt.xlabel("time step")

    #         plt.subplot(2, 1, 2)
    #         plt.ylim(-5000,0)
    #         plt.plot(self.total_rewards)
    #         plt.title('Total Rewards')
    #         plt.ylabel("Reward")
    #         plt.xlabel("eps")

    #         plt.tight_layout()

    #         if(plot == 5):
    #             plt.savefig(location + "TEST_" + str(timestep) + "_REWARDS", dpi=300)
    #         else:
    #             plt.savefig(location + str(timestep) + "_REWARDS", dpi=300)
            
    #         plt.clf() 


    def graph(self, location):
        for plot, dump in enumerate(self.log_dumps[-1:]):
            settings, depths, floodings = [[] for i in range(self.R)], [[] for i in range(
                self.R + self.J - 1)], [[] for i in range(self.R + self.J - 1)]
            rewards = []

            for reward, state, timestep in dump:  # timestep should stay local here

                rewards.append(reward)
                setting, depth, flooding = state

                for i, R in enumerate(settings):
                    R.append(setting[i])
                for i, S_depth in enumerate(depths):
                    S_depth.append(depth[i])
                for i, S_flooding in enumerate(floodings):
                    S_flooding.append(flooding[i])

            flood = sum([sum(floodings[i]) for i in range(len(floodings))])

            # print("Total Flooding:", flood)

            return flood
