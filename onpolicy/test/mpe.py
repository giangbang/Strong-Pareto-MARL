import gym
from onpolicy.envs.mpe.MPE_env import MPEEnv

if __name__ == "__main__":
    args = {
        "scenario_name": "simple_spread",
        "episode_length": 25,
        "num_agents": 2,
        "num_landmarks": 3
    }
    from argparse import Namespace
    args = Namespace(**args)
    env = MPEEnv(args)
    print(env.reset()[0].shape)
    ac = []
    for i in range(2):
        ac.append(env.action_space[i].sample())
    import numpy as np
    ac = np.array(ac)
    ac = np.eye(env.action_space[0].n)[ac]
    print(ac)
    print(env.step(ac))
    print(env.share_observation_space)
    print(env.observation_space)
    # reward shape: n_agent x 1
    # done shape: n_agent