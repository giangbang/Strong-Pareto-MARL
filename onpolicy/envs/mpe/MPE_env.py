from onpolicy.envs.mpe.environment import MultiAgentEnv
from onpolicy.envs.mpe.scenarios import load


def MPEEnv(args):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info)
    return env

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

    print(env.reset())
    ac = []
    for i in range(2):
        ac.append(env.action_space[i].sample())
    import numpy as np
    ac = np.array(ac)[None, ...]
    print(ac)
    print(env.step(ac))