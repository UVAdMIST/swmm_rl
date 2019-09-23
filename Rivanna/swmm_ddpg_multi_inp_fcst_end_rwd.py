# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 03:33:41 2019

@author: cw8xk
"""

import os
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import sys
sys.path.insert(0, '/home/bdb3m/swmm_rl')
from swmm_Model_multi_inp_fcst_end_rwd import BasicEnv

depth = 4.61  # maximum depth of ponds
repeats = 7  # number of consecutive times to train on single .inp file
data_dir = '/home/bdb3m/swmm_rl'
start_time = datetime.now()

# loop through training envs
file_num = 1
for file in os.scandir(os.path.join(data_dir, "syn_inp_train")):
    if file.name.endswith('.inp'):
        # print(file.name)
        # get forecast file that corresponds to current inp file
        fcst = os.path.join(data_dir, "fcst_train", file.name.split('.')[0] + ".csv")
        env = BasicEnv(inp_file=file.path, fcst_file=fcst, depth=depth)
        train_steps = env.T * repeats
        if file_num == 1:
            assert len(env.action_space.shape) == 1
            nb_actions = env.action_space.shape[0]
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(42)
            rn.seed(12345)
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            tf.set_random_seed(1234)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)

            actor = Sequential()
            actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
            actor.add(Dense(16))
            actor.add(Activation('relu'))
            actor.add(Dense(16))
            actor.add(Activation('relu'))
            # actor.add(Dense(8))
            # actor.add(Activation('relu'))
            actor.add(Dense(nb_actions))
            actor.add(Activation('sigmoid'))
            #print(actor.summary())

            action_input = Input(shape=(nb_actions,), name='action_input')
            observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
            flattened_observation = Flatten()(observation_input)
            x = Concatenate()([action_input, flattened_observation])
            x = Dense(32)(x)
            x = Activation('relu')(x)
            x = Dense(32)(x)
            x = Activation('relu')(x)
            x = Dense(32)(x)
            x = Activation('relu')(x)
            x = Dense(1)(x)
            x = Activation('linear')(x)
            critic = Model(inputs=[action_input, observation_input], outputs=x)
            # print(critic.summary())

            memory = SequentialMemory(limit=1000000, window_length=1)
            random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
            agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                              memory=memory, nb_steps_warmup_critic=50, nb_steps_warmup_actor=50,
                              random_process=random_process, gamma=.99, target_model_update=1e-3)
            agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

            agent.fit(env, nb_steps=train_steps, verbose=0)
            agent.save_weights('/home/bdb3m/swmm_rl/agent_weights_fcst/ddpg_swmm_weights.h5f', overwrite=True)
            env.close()

        else:
            agent.load_weights('/home/bdb3m/swmm_rl/agent_weights_fcst/ddpg_swmm_weights.h5f')
            agent.fit(env, nb_steps=train_steps, verbose=0)
            agent.save_weights('/home/bdb3m/swmm_rl/agent_weights_fcst/ddpg_swmm_weights.h5f', overwrite=True)
            env.close()

        if file_num % 1000 == 0:
            print("finished training on ", file_num, " files")
        file_num += 1

# loop through testing envs
for file in os.scandir(os.path.join(data_dir, "syn_inp_test")):
    if file.name.endswith('.inp'):
        print('testing ', file.name)
        fcst = os.path.join(data_dir, "fcst_test", file.name.split('.')[0] + ".csv")
        env = BasicEnv(inp_file=file.path, fcst_file=fcst, depth=depth)
        history = agent.test(env, nb_episodes=1, visualize=False, nb_max_start_steps=0)
        env.close()

        # get rain/tide data from inp file
        rain_str = []
        tide_str = []
        with open(file.path, 'r') as tmp_file:
            lines = tmp_file.readlines()
            for i, l in enumerate(lines):
                if l.startswith("[TIMESERIES]"):  # find time series section
                    start = i + 3
        for i, l in enumerate(lines[start:]):
            if l.startswith('Atlas14'):
                rain_str.append(l)
            if l.startswith('Tide1'):
                tide_str.append(l)

        rain_data = []
        rain_time = []
        tide_data = []
        tide_time = []
        for i in rain_str:
            rain_data.append(i.split(' ')[3].rstrip())
            rain_time.append(i.split(' ')[1] + " " + i.split(' ')[2])

        for i in tide_str:
            tide_data.append(i.split(' ')[3].rstrip())
            tide_time.append(i.split(' ')[1] + " " + i.split(' ')[2])

        rain_df = pd.DataFrame([rain_time, rain_data]).transpose()
        rain_df.columns = ['datetime', 'rain']
        rain_df['datetime'] = pd.to_datetime(rain_df['datetime'], infer_datetime_format=True)
        rain_df.set_index(pd.DatetimeIndex(rain_df['datetime']), inplace=True)
        rain_df['rain'] = rain_df['rain'].astype('float')
        rain_df = rain_df.resample('H').mean()

        tide_df = pd.DataFrame([tide_time, tide_data], dtype='float64').transpose()
        tide_df.columns = ['datetime', 'tide']
        tide_df['datetime'] = pd.to_datetime(tide_df['datetime'], infer_datetime_format=True)
        tide_df.set_index(pd.DatetimeIndex(tide_df['datetime']), inplace=True)
        tide_df['tide'] = tide_df['tide'].astype('float')

        df = pd.concat([rain_df['rain'], tide_df['tide']], axis=1)
        df['rain'].fillna(0, inplace=True)
        df.reset_index(inplace=True)

        all_actions = np.array(history.history['action'])
        all_states = np.array(history.history['states'])
        all_depths = all_states[:, :, :3]
        all_flooding = all_states[:, :, 3:6]
        st_max = [depth] * len(all_depths[0])
        j3_max = [2] * len(all_depths[0])

        # # plot average rewards per episode
        # avg_reward = []
        # num_episodes = int(memory.nb_entries/env.T)
        #
        # for i in range(num_episodes):
        #     temp_rwd = memory.rewards.data[env.T * i: env.T * i + env.T]
        #     avg_reward.append(np.mean(temp_rwd))

        # plot results from test with learned policy
        fig, axs = plt.subplots(4, sharey='none', sharex='none', figsize=(6, 8))
        # first plot is tide and rainfall
        ax = axs[0]
        df["tide"].plot(ax=ax, color='c', legend=None)
        ax2 = ax.twinx()
        ax2.invert_yaxis()
        df["rain"].plot.bar(ax=ax2, color="b", legend=None)
        ax2.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_ylabel("Sea Level (ft.)")
        ax2.set_ylabel("Rainfall (in.)")
        ax.set_title('Inputs')
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, ("Sea Level", "Cumulative Rainfall"), bbox_to_anchor=(0.85, -0.05), ncol=2,
                  frameon=False)

        # plot depths
        depth_plot = axs[1].plot(all_depths[0])
        max_1 = axs[1].plot(st_max, linestyle=':', color='k', label='Storage max')
        max_2 = axs[1].plot(j3_max, linestyle=':', color='grey', label='Pipe max')
        axs[1].set_ylim(0, 6)
        axs[1].set_title('Depths')
        axs[1].set_ylabel("ft")
        # plt.xlabel("time step")
        # lines, labels = axs[1].get_legend_handles_labels()
        # axs[1].legend(lines, labels, bbox_to_anchor=(0.8, -0.05), ncol=5)
        first_legend = axs[1].legend(depth_plot, ('St1', 'St2', 'J1'), bbox_to_anchor=(0.0, -.5, 1., .102),
                                     loc=3, ncol=3, borderaxespad=0.1, frameon=False, columnspacing=.75)
        legend_ax = axs[1].add_artist(first_legend)
        axs[1].legend(loc=4, bbox_to_anchor=(0.025, -.5, 1., .11), ncol=2,
                      borderaxespad=0.1, frameon=False, columnspacing=.75)

        # plot actions
        act_plot = axs[2].plot(all_actions[0])
        axs[2].set_ylim(0, 1.05)
        axs[2].set_title('Policy')
        axs[2].set_ylabel("Valve Position")
        axs[2].legend(act_plot, ('R1', 'R2'))

        # plot flooding
        axs[3].plot(all_flooding[0])
        axs[3].set_ylim(0)
        axs[3].set_title('Flooding')
        axs[3].set_ylabel("CFS")
        axs[3].set_xlabel("time step")

        # plot average reward
        # plt.subplot(4, 1, 4)
        # plt.plot(avg_reward, color='k')
        # plt.ylim(plt.ylim()[0], 0)
        # plt.title('Average reward per episode')
        # plt.ylabel("reward")
        # plt.xlabel("episode")

        plt.tight_layout()
        # plt.show()
        plt.savefig("/home/bdb3m/swmm_rl/plots_fcst/" + file.name.split('.')[0] + ".png",
                    dpi=300)
        plt.close()

print('Total run time was {0}'.format(datetime.now() - start_time))
