try:
    import sys
    from StormwaterEnv import StormwaterEnv
    # print("Running on", sys.argv[1])
    # env = StormwaterEnv(R=2, J=2, J_offset=0, swmm_inp="25yr24-hour-10yrmax_0.inp")
    env = StormwaterEnv(R=2, J=2, J_offset=0, swmm_inp=sys.argv[1])

    env.reset()

    action = [0, 0.1]


    for i in range(95):
        obs, reward, done, debug = env.step(action)
        # action  = [depth/30 for depth in obs[6:12]]
        # action = [act - obs[12]/45 - obs[13]/45 for act in action]
        # action[-2:] = [1,1]
        # print(env.render())
        # if i > 44:
        #     action = [0, 0, 0, 0, 1, 1]

    # print(action1, action2)
    flood = env.graph("plots/baseline_")

    env.close()

    if flood != 0:
        print("Running on", sys.argv[1], flood)
except:
    print("Failed on", sys.argv[1])
