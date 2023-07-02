import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical, Normal, Uniform
from amb.models.base.cnn import CNNLayer
from amb.models.base.mlp import MLPBase
from amb.models.base.rnn import RNNLayer
from amb.utils.env_utils import (
    check,
    get_shape_from_obs_space,
    get_onehot_shape_from_act_space,
)
from amb.utils.model_utils import get_active_func, init, get_init_method


class QActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(QActor, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.args = args
        self.hidden_sizes = args["hidden_sizes"]
        self.activation_func = args["activation_func"]
        self.final_activation_func = args["final_activation_func"]
        self.initialization_method = args["initialization_method"]
        self.expl_noise = args["expl_noise"]

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        init_method = get_init_method(self.initialization_method)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.act_shape = get_onehot_shape_from_act_space(action_space)

        if len(obs_shape) == 3:
            self.cnn = CNNLayer(
                obs_shape,
                self.hidden_sizes,
                self.initialization_method,
                self.activation_func,
            )
            input_dim = self.cnn.output_size
        else:
            self.cnn = nn.Identity()
            input_dim = obs_shape[0]

        self.base = MLPBase(args, input_dim)

        if self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.out = init_(nn.Linear(self.hidden_sizes[-1], self.act_shape))

        self.action_type = action_space.__class__.__name__

        self.to(device)

    def sample(self, obs, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.action_type == "Box":
            raise f"Box action space is not supported for {self.__class__.__name__}"
        elif self.action_type == "Discrete" and available_actions is not None:
            q_values = torch.ones((obs.shape[0], self.act_shape)).to(**self.tpdv)
            q_values[available_actions == 0] = -1e10

        return q_values

    def forward(self, obs, rnn_states, masks, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(self.cnn(obs))
        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        q_values = self.out(actor_features)

        if self.action_type == "Box":
            raise f"Box action space is not supported for {self.__class__.__name__}"
        elif self.action_type == "Discrete" and available_actions is not None:
            q_values[available_actions == 0] = -1e10

        return q_values, rnn_states
