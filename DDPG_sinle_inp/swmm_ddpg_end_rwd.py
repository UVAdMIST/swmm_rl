# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 03:33:41 2019

@author: cw8xk
"""
import os
import random as rn
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from swmm_keras_rl.swmm_Model_end_rwd import BasicEnv

Depth = 4.61
env = BasicEnv(depth=Depth)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# np.random.seed(123)
# set_random_seed(1234)

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
train_steps = 30000
# agent.load_weights('ddpg_swmm_weights_{}_depth_{}.h5f'.format(train_steps,Depth))
# agent.load_weights('ddpg_swmm_weights_{}.h5f'.format(train_steps))

agent.fit(env, nb_steps=train_steps, verbose=1)
agent.save_weights('ddpg_swmm_weights_{}_depth_{}.h5f'.format(train_steps,Depth), overwrite=True)

history = agent.test(env, nb_episodes=1, visualize=False, nb_max_start_steps=0)
all_actions = np.array(history.history['action'])
all_states = np.array(history.history['states'])
all_depths = all_states[:, :, :3]
all_flooding = all_states[:, :, 3:]
st_max = [Depth] * len(all_depths[0])
j3_max = [2] * len(all_depths[0])

# plot average rewards per episode
avg_reward = []
num_episodes = int(memory.nb_entries/env.T)

for i in range(num_episodes):
    temp_rwd = memory.rewards.data[env.T * i: env.T * i + env.T]
    avg_reward.append(np.mean(temp_rwd))

# plot results from test with learned policy
fig = plt.figure(1, figsize=(6, 8))

plt.subplot(4, 1, 1)
depth_plot = plt.plot(all_depths[0])
max_1 = plt.plot(st_max, linestyle=':', color='k', label='Storage max')
max_2 = plt.plot(j3_max, linestyle=':', color='grey', label='Pipe max')
plt.ylim(0, 6)
plt.title('Depths')
plt.ylabel("ft")
# plt.xlabel("time step")
first_legend = plt.legend(depth_plot, ('St1', 'St2', 'J3'), bbox_to_anchor=(0.03, -.5, 1., .102), loc=3,
                          ncol=3, borderaxespad=0.1, frameon=False, columnspacing=1)
ax = plt.gca().add_artist(first_legend)
plt.legend(loc=4, bbox_to_anchor=(-0.025, -.5, 1., .102), ncol=2, borderaxespad=0.1, frameon=False, columnspacing=1)

plt.subplot(4, 1, 2)
act_plot = plt.plot(all_actions[0])
plt.ylim(-0.95, 1.05)
plt.title('Policy')
plt.ylabel("Valve Position")
plt.xlabel("time step")
plt.legend(act_plot, ('R1', 'R2'))

plt.subplot(4, 1, 3)
plt.plot(all_flooding[0],  label=['St1', 'St2', 'J3'])
plt.ylim(0)
plt.title('Flooding')
plt.ylabel("CFS")
# plt.xlabel("time step")

plt.subplot(4, 1, 4)
plt.plot(avg_reward, color='k')
plt.ylim(plt.ylim()[0], 0)
plt.title('Average reward per episode')
plt.ylabel("reward")
plt.xlabel("episode")

plt.tight_layout()
# plt.show()
plt.savefig("C:/Users/Ben Bowes/PycharmProjects/swmm_keras_rl/ddpg_" + str(train_steps) + "steps_end_rwd.png", dpi=300)
# plt.close()

# # Cheng's plotting code
# plt.figure(0)
# plt.plot(all_actions[0,:,0],label = 'R1')
# plt.plot(all_actions[0,:,1],'-.', label = 'R2')
# plt.grid(True)
# plt.legend(loc=0)
# plt.ylim(-0.1,1.1)
# plt.title('Actions After {} Training Steps, depth={}'.format(train_steps,Depth))
# plt.savefig('Actions_{}_depth_{}.pdf'.format(train_steps,Depth),dpi=300)
#
# total_flooding = np.sum(all_floodings,axis=2)
# total_flooding = np.cumsum(total_flooding[0])
# plt.figure(1)
# plt.plot(total_flooding)
# plt.grid(True)
# plt.title('Total Flooding After {} Training Steps, depth = {}'.format(train_steps,Depth))
# plt.savefig('Flooding_{}_depth_{}.pdf'.format(train_steps,Depth),dpi=300)
# #plt.savefig('Flooding_{}_depth_{}_test.pdf'.format(train_steps,Depth),dpi=300)
#
# plt.figure(2)
# plt.plot(avg_reward, label='avg_rwd')
# plt.grid(True)
# plt.legend(loc=0)
# plt.title('Average Reward per Episode After {} Training Steps, depth={}'.format(train_steps, Depth))
# plt.savefig('Avg_Epsi_Rwds_{}_depth_{}.pdf'.format(train_steps, Depth), dpi=300)
