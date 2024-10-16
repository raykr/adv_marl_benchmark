import torch
import numpy as np
from amb.data.episode_buffer import EpisodeBuffer
from amb.runners.dual.base_runner import BaseRunner
from amb.utils.popart import PopArt
from amb.utils.trans_utils import _t2n
from amb.utils.env_utils import (
    get_shape_from_obs_space,
    get_shape_from_act_space,
)


class OnPolicyRunner(BaseRunner):
    def __init__(self, args, algo_args, env_args):
        """Initialize the dual/OnPolicyRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        super(OnPolicyRunner, self).__init__(args, algo_args, env_args)

        if self.algo_args["angel"]['use_render'] is False:  # train, not render
            self.buffers = []
            for agent_id in range(self.num_angels):
                scheme = {
                    "obs": {"vshape": get_shape_from_obs_space(self.envs.observation_space[0][agent_id]), "offset": 1},
                    "rnn_states_actor": {"vshape": (self.angel_recurrent_n, self.angel_rnn_hidden_size), "offset": 1, "extra": ["rnn_state"]},
                    "share_obs": {"vshape": get_shape_from_obs_space(self.envs.share_observation_space[0][0]), "offset": 1},
                    "rnn_states_critic": {"vshape": (self.angel_recurrent_n, self.angel_rnn_hidden_size), "offset": 1, "extra": ["rnn_state"]},
                    "actions": {"vshape": (get_shape_from_act_space(self.envs.action_space[0][agent_id]),), "offset": 0},
                    "action_log_probs": {"vshape": (get_shape_from_act_space(self.envs.action_space[0][agent_id]),), "offset": 0},
                    "value_preds": {"vshape": (1,), "offset": 0, "extra": ["more_length"]},
                    "rewards": {"vshape": (1,), "offset": 0},
                    "returns": {"vshape": (1,), "offset": 0, "extra": ["more_length"]},
                    "advantages": {"vshape": (1,), "offset": 0},
                    "masks": {"vshape": (1,), "offset": 1, "init_value": 1},
                    "active_masks": {"vshape": (1,), "offset": 1, "init_value": 1},
                    "bad_masks": {"vshape": (1,), "offset": 1, "init_value": 1},
                }
                if self.action_type == "Discrete":
                    scheme["available_actions"] = {"vshape": (self.envs.action_space[0][agent_id].n,), "offset": 1, "init_value": 1}
                self.buffers.append(EpisodeBuffer(algo_args["angel"], self.n_rollout_threads, scheme))

            if self.algo_args["angel"]["use_popart"] is True:
                self.value_normalizer = PopArt(1, device=self.device)
            else:
                self.value_normalizer = None

        self.restore()

    def init_batch(self):
        """initialize the replay buffer."""
        obs, share_obs, available_actions = self.envs.reset()
        for agent_id in range(self.num_angels):
            data = {
                "obs": obs[0][:, agent_id].copy(),
                "share_obs": share_obs[0][:, agent_id].copy()
            }
            if "available_actions" in self.buffers[agent_id].data:
                data["available_actions"] = available_actions[0][:, agent_id].copy()
            self.buffers[agent_id].init_batch(data)
        return obs, share_obs, available_actions

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["angel"]['use_render'] is True:
            self.render()
            return
        print("start running")
        self.logger.init()  # logger callback at the beginning of training

        self.logger.episode_init(0) 
        self.eval()

        obs, share_obs, available_actions = self.init_batch()
        demon_rnn_states = np.zeros((self.n_rollout_threads, self.num_demons, self.demon_recurrent_n, self.demon_rnn_hidden_size), dtype=np.float32)
        demon_masks = np.ones((self.n_rollout_threads, self.num_demons, 1), dtype=np.float32)

        episodes = int(self.algo_args["angel"]['num_env_steps']) // self.algo_args["angel"]['episode_length'] // self.algo_args["angel"]['n_rollout_threads']
        
        for episode in range(1, episodes + 1):
            if self.algo_args["angel"]['use_linear_lr_decay']:  # linear decay of learning rate
                self.algo.lr_decay(episode, episodes)

            self.logger.episode_init(episode * self.algo_args["angel"]['episode_length'] * self.algo_args["angel"]['n_rollout_threads'])  # logger callback at the beginning of each episode

            self.algo.prep_rollout()

            for step in range(self.algo_args["angel"]['episode_length']):
                # Sample actions from actors and values from critics
                values, angel_actions, action_log_probs, angel_rnn_states, rnn_states_critic = self.collect(step)

                demon_actions_collector = []
                for agent_id in range(self.num_demons):
                    demon_actions, temp_rnn_state = self.demons[agent_id].perform(
                        obs[1][:, agent_id],
                        demon_rnn_states[:, agent_id],
                        demon_masks[:, agent_id],
                        available_actions[1][:, agent_id]
                        if available_actions[1][0] is not None else None,
                        deterministic=False
                    )
                    demon_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    demon_actions_collector.append(_t2n(demon_actions))
                demon_actions = np.stack(demon_actions_collector, axis=1)

                # actions: (n_threads, n_agents, action_dim)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step((angel_actions, demon_actions))

                filled = np.ones((self.n_rollout_threads, self.num_angels), dtype=np.float32)

                dones_env = np.all(dones[1], axis=1)
                demon_rnn_states[dones_env==True] = 0
                demon_masks = np.ones((self.n_rollout_threads, self.num_demons, 1), dtype=np.float32)
                demon_masks[dones_env==True] = 0

                data = {
                    "obs": obs[0], "share_obs": share_obs[0], "rewards": rewards[0], "dones": dones[0],
                    "infos": infos[0], "value_preds": values, "actions": angel_actions, "action_log_probs": action_log_probs,
                    "rnn_states_actor": angel_rnn_states, "rnn_states_critic": rnn_states_critic, "filled": filled
                }
                if "available_actions" in self.buffers[0].data:
                    data.update({"available_actions": available_actions[0]})
                
                self.logger.per_step(data)  # logger callback at each step
                self.insert(data, step)  # insert data into buffer

            # compute return and update network
            value_collector = []
            for agent_id in range(self.num_angels):
                value, _ = self.critic(
                    self.buffers[agent_id].data["share_obs"][:, step],
                    self.buffers[agent_id].data["rnn_states_critic"][:, step],
                    self.buffers[agent_id].data["masks"][:, step],
                )
                value_collector.append(_t2n(value))
            next_values = np.stack(value_collector, axis=1)

            self.algo.prep_training()
            actor_train_infos, critic_train_info = self.train(next_values)

            # log information
            if episode % self.algo_args["angel"]['log_interval'] == 0:
                self.logger.episode_log(actor_train_infos, critic_train_info, self.buffers)

            # eval
            if episode % self.algo_args["angel"]['eval_interval'] == 0:
                if self.algo_args["angel"]['use_eval']:
                    self.eval()
                self.save()

            for buffer in self.buffers:
                buffer.after_update()

    @torch.no_grad()
    def collect(self, step):
        """Collect actions and values from actors and critics."""
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        value_collector = []
        rnn_state_critic_collector = []

        for agent_id in range(self.num_angels):
            action, action_log_prob, rnn_state = self.angels[agent_id].collect(
                self.buffers[agent_id].data["obs"][:, step],
                self.buffers[agent_id].data["rnn_states_actor"][:, step],
                self.buffers[agent_id].data["masks"][:, step],
                self.buffers[agent_id].data["available_actions"][:, step]
                if "available_actions" in self.buffers[agent_id].data else None
            )
            value, rnn_state_critic = self.critic(
                self.buffers[agent_id].data["share_obs"][:, step],
                self.buffers[agent_id].data["rnn_states_critic"][:, step],
                self.buffers[agent_id].data["masks"][:, step],
            )
            
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            value_collector.append(_t2n(value))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))

        actions = np.stack(action_collector, axis=1)
        action_log_probs = np.stack(action_log_prob_collector, axis=1)
        rnn_states = np.stack(rnn_state_collector, axis=1)
        values = np.stack(value_collector, axis=1)
        rnn_states_critic = np.stack(rnn_state_critic_collector, axis=1)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def insert(self, data, step):
        """Insert data into buffer.
           obs, share_obs, rewards, dones, infos, available_actions, values, 
           actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        dones_env = np.all(data["dones"], axis=1)
        data["rnn_states_actor"][dones_env==True] = 0
        data["rnn_states_critic"][dones_env==True] = 0

        data["masks"] = np.ones((self.n_rollout_threads, self.num_angels, 1), dtype=np.float32)
        data["masks"][dones_env==True] = 0

        data["active_masks"] = np.ones((self.n_rollout_threads, self.num_angels, 1), dtype=np.float32)
        data["active_masks"][data["dones"]==True] = 0
        data["active_masks"][dones_env==True] = 1

        data["bad_masks"] = np.ones((self.n_rollout_threads, self.num_angels, 1), dtype=np.float32)
        for i in range(self.n_rollout_threads):
            for j in range(self.num_angels):
                if "bad_transition" in data["infos"][i][j] and data["infos"][i][j]["bad_transition"] == True:
                    data["bad_masks"][i, j] = 0

        del data["infos"]
        del data["dones"]

        # print({k: data[k].shape for k in data})
        for agent_id in range(self.num_angels):
            self.buffers[agent_id].insert({k: data[k][:, agent_id] for k in data}, step)

    def train(self, next_values):
        """Training procedure for MAPPO."""
        advantages = []
        for agent_id in range(self.num_angels):
            self.buffers[agent_id].compute_returns(next_values[:, agent_id], self.value_normalizer)
            if self.value_normalizer is not None:
                advantage = self.buffers[agent_id].data["returns"][:, :-1] - self.value_normalizer.denormalize(self.buffers[agent_id].data["value_preds"])[:, :-1]
            else:
                advantage = self.buffers[agent_id].data["returns"][:, :-1] - self.buffers[agent_id].data["value_preds"][:, :-1]
            advantages.append(advantage)
        advantages = np.stack(advantages, axis=2)

        active_masks_collector = [self.buffers[i].data["active_masks"] for i in range(self.num_angels)]
        active_masks_array = np.stack(active_masks_collector, axis=2)
        advantages_copy = advantages.copy()
        advantages_copy[active_masks_array[:, :-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for agent_id in range(self.num_angels):
            self.buffers[agent_id]["advantages"][:] = advantages[:, :, agent_id].copy()

        # update actors
        actor_train_infos = []
        if self.angel_share_param:
            actor_train_info = self.algo.share_param_train_actor(self.buffers)
            for _ in range(self.num_angels):
                actor_train_infos.append(actor_train_info)
        else:
            for agent_id in range(self.num_angels):
                actor_train_info = self.algo.train_actor(self.buffers[agent_id], agent_id)
                actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.algo.train_critic(self.buffers, self.value_normalizer)

        return actor_train_infos, critic_train_info

    def save(self):
        """Save model parameters."""
        super().save()
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer.pth",
            )

    def restore(self):
        """Restore model parameters."""
        super().restore()
        if self.algo_args["angel"]['model_dir'] is not None:
            if self.algo_args["angel"]['use_render'] is False and self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_args["angel"]['model_dir']) + "/value_normalizer.pth"
                )
                self.value_normalizer.load_state_dict(value_normalizer_state_dict)