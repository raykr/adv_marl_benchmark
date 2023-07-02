import os
import torch
from amb.agents.base_agent import BaseAgent
from amb.models.actor.q_actor import QActor
from amb.utils.action_selectors import REGISTRY as action_REGISTRY


class QAgent(BaseAgent):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        # save arguments
        self.args = args
        self.device = device

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = QActor(args, self.obs_space, self.act_space, self.device)
        self.action_selector = action_REGISTRY[self.args["action_selector"]](self.args)
        self.current_timestep = 0

    def forward(self, obs, rnn_states, masks, available_actions=None):
        q_values, rnn_states = self.actor(obs, rnn_states, masks, available_actions)

        # epsion-greedy policy to select action
        actions, _ = self.action_selector.select(q_values, available_actions, self.current_timestep, device=self.device)

        return actions, rnn_states

    @torch.no_grad()
    def perform(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        q_values, rnn_states = self.actor(obs, rnn_states, masks, available_actions)

        # argmax Q-values to select action
        actions = q_values.argmax(dim=-1, keepdim=True)

        return actions, rnn_states

    @torch.no_grad()
    def sample(self, obs, available_actions=None):
        q_values = self.actor.sample(obs, available_actions)

        # epsion-greedy policy to select action
        actions, actions_onehot = self.action_selector.select(q_values, available_actions, self.current_timestep, device=self.device)

        return actions, actions_onehot

    @torch.no_grad()
    def collect(self, obs, rnn_states, masks, available_actions=None):
        q_values, rnn_states = self.actor(obs, rnn_states, masks, available_actions)

        # epsion-greedy policy to select action
        actions, actions_onehot = self.action_selector.select(q_values, available_actions, self.current_timestep, device=self.device)

        return actions, actions_onehot, rnn_states

    def restore(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))

    def prep_training(self):
        self.actor.train()

    def prep_rollout(self):
        self.actor.eval()
