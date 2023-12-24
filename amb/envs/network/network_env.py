
import os

import numpy as np
from amb.envs.network.envs.large_grid_env import LargeGridEnv
import configparser
from gym.spaces import Discrete

from amb.envs.network.envs.real_net_env import RealNetEnv


class NetworkEnv:
    def __init__(self, env_args) -> None:
        ncfg = env_args["network_cfg"]
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "config", ncfg))

        self.env = LargeGridEnv(config["ENV_CONFIG"], 2, env_args["output_dir"], is_record=True, record_stat=True)
        self.env.reset()
        self.n_agents = self.env.n_agent
        self.n_s_ls = self.env.n_s_ls
        self.n_a_ls = self.env.n_a_ls
        print(self.n_s_ls, self.n_a_ls)
        # if (max(self.n_s_ls) == min(self.n_s_ls)):
        #     # note for identical IA2C, n_s_ls may have varient dims
        #     self.identical_agent = True
        #     self.n_s = self.n_s_ls[0]
        #     self.n_a = self.n_a_ls[0]
        # else:
        #     self.n_s = max(self.n_s_ls)
        #     self.n_a = max(self.n_a_ls)
        self.observation_space = np.array(self.n_s_ls).reshape(len(self.n_s_ls), 1).tolist()
        self.share_observation_space = np.array(self.n_s_ls).reshape(len(self.n_s_ls), 1).tolist()
        self.action_space = [Discrete(value) for value in self.n_a_ls]
        print(f"n_agents: {self.n_agents}")
        print(f"action space: {self.action_space}")
        print(f"observation space: {self.observation_space}")
        print(f"share_observation_space: {self.share_observation_space}")
        print(f"obs class", type(self.observation_space.__class__.__name__))
        print(f"available_actions: ", self.get_avail_actions())


    def step(self, actions):
        """Process a step of the environment.
        actions: numpy.ndarray (num_agents, action_shape). Actions must be 2-dimentional.

        obs, share_obs: numpy.ndarray (num_agents, vshape)
        rewards: numpy.ndarray (num_agents, 1). Rewards for different agents can be different.
        dones: boolean numpy.ndarray (num_agents, 1). True when an episode is done or the time is out of limit, else False.
        infos: list of dict, e.g., [{}, {}]
        available_actions: 0-1 numpy.ndarray (num_agents, action_num) or None.
        """
        obs, reward, done, global_reward = self.env.step(actions)
        rewards = np.array([global_reward] * self.n_agents)
        dones = np.array([done] * self.n_agents)
        infos = [{}] * self.n_agents
        return obs, obs, rewards, dones, infos, self.get_avail_actions()

    def reset(self):
        obs = self.env.reset()
        return obs, obs, self.get_avail_actions()

    def seed(self, seed):
        self.env.seed = seed
        self.env.reset()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n
