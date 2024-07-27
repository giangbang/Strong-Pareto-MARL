import gym
import numpy as np
import torch as th

payoff_values = [[(1, 2), (0, 2)],
                [(1, 0), (0, 0)]]



class OneStepMatrixGame():
    def __init__(self, batch_size=None, **kwargs):
        # Define the agents
        self.n_agents = 2

        # Define the internal state
        self.steps = 0
        self.n_actions = len(payoff_values[0])
        self.episode_limit = 1
        self.n_agents = 2
        self.observation_space = []
        for i in range(self.n_agents):
            self.observation_space.append(gym.spaces.Box(
                low=0, high=1, shape=(2,), dtype=np.float32))
        self.share_observation_space = [gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32) for _ in range(self.n_agents)]
        self.action_space = [gym.spaces.Discrete(2)]*self.n_agents


    def reset(self):
        """ Returns initial observations and states"""
        self.steps = 0
        return self.get_obs()

    def step(self, actions):
        """ Returns reward, terminated, info """
        print(self.action_space)
        actions = np.array(actions, dtype=int)
        print("action", actions)
        reward = payoff_values[actions[0]][actions[1]]

        self.steps = 1
        terminated = [True]*len(reward)

        info = {}
        return self.get_obs(), reward, terminated, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        one_hot_id = np.zeros((2, 2), dtype=bool)
        one_hot_id[0, 0] = 1
        one_hot_id[1, 1] = 1
        return one_hot_id[0], one_hot_id[1]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_state(self):
        return self.get_obs_agent(0)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, s):
        pass
    
    def get_env_info(self):
        info = super().get_env_info()
        info.update({"agent_features": []})
        return info
    
    def get_visibility_matrix(self):
        return np.ones((self.n_agents, self.n_agents), dtype=bool)
    
if __name__ == "__main__":
    env = OneStepMatrixGame()
    print(env.reset())
    print(env.step([0, 1]))