import onpolicy.envs.rware 
import gym

if __name__ == "__main__":
    env = gym.make("rware-small-4ag-v1")
    print(env.reset())
    action = env.action_space.sample()
    print( env.step(action) )
    print(env.action_space)
    # reward shape: n_agent
    # done shape: n_agent