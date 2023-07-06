
from copy import deepcopy
import torch
from amb.algorithms.qmix import QMIX
from amb.models.critic.vdn_mixer import VDNMixer

class VDN(QMIX):
    def __init__(self, args, num_agents, obs_spaces, share_obs_space, act_spaces, device):
        super().__init__(args, num_agents, obs_spaces, share_obs_space, act_spaces, device)

        self.critic = VDNMixer()
        self.target_critic = deepcopy(self.critic)
        self.params += list(self.critic.parameters())

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        self._off_grad()