from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from stormwater.utils.rl.ddpg import DDPGAgent
from stormwater.utils.rl.memory import SequentialMemory
from stormwater.utils.rl.rand import OrnsteinUhlenbeckProcess

# from rl.agents import DDPGAgent
# from rl.memory import SequentialMemory
# from rl.random import OrnsteinUhlenbeckProcess

#import Env as enviornment
from stormwater.envs.StormwaterEnv import StormwaterEnv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
This script runs Deep Q-Network RL algorithm for control
of stormwater systems using a SWMM model as the environment
"""

env = StormwaterEnv()
nb_actions = 6

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + (24,)))
actor.add(Dense(10))
actor.add(Activation('relu'))
actor.add(Dense(10))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + (24,), name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(10)(x)
x = Activation('relu')(x)
x = Dense(10)(x)
x = Activation('relu')(x)
x = Dense(10)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)

memory = SequentialMemory(limit=1000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=300,
                  random_process=random_process, gamma=.995, target_model_update=2e-3)
agent.compile(Adam(lr=.005, clipnorm=1.), metrics=['mae'])

agent.fit(env, nb_steps=95, visualize=False, verbose=0, nb_max_episode_steps=95)

agent.save_weights('weights/ddpg_{}_weights.h5f'.format("stormwater"), overwrite=True)

agent.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=95)


# env.graph("../plots/test_plot_")