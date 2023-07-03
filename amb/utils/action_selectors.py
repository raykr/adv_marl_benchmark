import torch as th
from torch.distributions import Categorical
import numpy as np


class DecayThenFlatSchedule:
    def __init__(self, start, finish, time_length, decay="exp"):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (
                (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1
            )

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(-T / self.exp_scaling)))

    pass


class ActionSelectorBase:
    def __init__(self, args):
        self.args = args

    def select(self, agent_inputs, avail_actions, t_env, test_mode=False):
        raise NotImplementedError


class MultinomialActionSelector(ActionSelectorBase):
    def __init__(self, args):
        super(MultinomialActionSelector, self).__init__(args)

        self.schedule = DecayThenFlatSchedule(
            args["epsilon_start"],
            args["epsilon_finish"],
            args["epsilon_anneal_time"],
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


class EpsilonGreedyActionSelector(ActionSelectorBase):
    def __init__(self, args):
        super(EpsilonGreedyActionSelector, self).__init__(args)

        self.schedule = DecayThenFlatSchedule(
            args["epsilon_start"],
            args["epsilon_finish"],
            args["epsilon_anneal_time"],
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)

    def select(self, agent_inputs, avail_actions, t_env, test_mode=False, device="cpu"):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone().to(device)
        # masked_q_values[avail_actions == 0] = -1e10

        random_numbers = th.rand_like(agent_inputs[:, 0]).to(device)
        pick_random = (random_numbers < self.epsilon).long().to(device)
        random_actions = Categorical(th.from_numpy(avail_actions)).sample().long().to(device)

        picked_actions = (
            pick_random * random_actions
            + (1 - pick_random) * masked_q_values.max(dim=1)[1]
        )
        picked_actions = picked_actions.unsqueeze(-1).to(device)

        # one-hot
        picked_actions_onehot = th.zeros_like(agent_inputs).to(device)
        picked_actions_onehot.scatter_(-1, picked_actions, 1.0)

        return picked_actions, picked_actions_onehot


class SoftPoliciesSelector(ActionSelectorBase):
    def __init__(self, args):
        super(EpsilonGreedyActionSelector, self).__init__(args)

    def select(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions


REGISTRY = {
    "epsilon_greedy": EpsilonGreedyActionSelector,
    "multinomial": MultinomialActionSelector,
    "soft_policies": SoftPoliciesSelector,
}
