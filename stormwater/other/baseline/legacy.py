from StormwaterEnv import StormwaterEnv

env = StormwaterEnv()

env.reset()

# action = [0 for i in range(6)]
action = [0, 0, 0, 0, 1, 1]

for i in range(95):
    obs, reward, done, debug = env.step(action)
    # action  = [depth/30 for depth in obs[6:12]]
    # action = [act - obs[12]/45 - obs[13]/45 for act in action]
    # action[-2:] = [1,1]
    print(env.render())
    # if i > 44:
    #     action = [0, 0, 0, 0, 1, 1]


env.graph("plots/baseline_")
env.close()