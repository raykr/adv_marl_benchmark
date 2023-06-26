import os
import time
import torch
import numpy as np
import setproctitle
from amb.algorithms import ALGO_REGISTRY
from amb.envs import LOGGER_REGISTRY
from amb.utils.trans_utils import _t2n
from amb.utils.env_utils import (
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from amb.utils.model_utils import init_device
from amb.utils.config_utils import init_dir, save_config, get_task_name


class BaseRunner:
    def __init__(self, args, algo_args, env_args):
        """Initialize the perturbation/BaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.hidden_sizes = algo_args["victim"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["victim"]["recurrent_n"]
        self.share_param = algo_args["victim"]['share_param']

        self.n_rollout_threads = algo_args["train"]["n_rollout_threads"]

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.task_name = get_task_name(args["env"], env_args)
        if not self.algo_args['render']['use_render']:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["victim"],
                args["exp_name"],
                args["run"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"]) + "-" + "perturbation"
        )

        # set the config of env
        if self.algo_args['render']['use_render']:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        self.action_type = self.envs.action_space[0].__class__.__name__

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space, self.action_type)

        if self.algo_args['render']['use_render'] is False:
            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )

        # algorithm
        if self.share_param:
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id] == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."

        self.agents = []
        self.attacks = []
        if self.share_param:
            agent = ALGO_REGISTRY[args["victim"]].create_agent(
                algo_args["victim"],
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            attack = ALGO_REGISTRY[args["algo"]](algo_args["attack"], self.envs.action_space[0], self.device)
            for agent_id in range(self.num_agents):
                self.agents.append(agent)
                self.attacks.append(attack)
        else:
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["victim"]].create_agent(
                    algo_args["victim"],
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                attack = ALGO_REGISTRY[args["algo"]](algo_args["attack"], self.envs.action_space[agent_id], self.device)
                self.agents.append(agent)
                self.attacks.append(attack)

        if self.algo_args['victim']['model_dir'] is not None:  # restore model
            self.restore()

    def run(self):
        print("start perturbation-based attack")
        self.logger.episode_init(0)
        self.eval()
        self.eval_adv()

    @torch.no_grad()
    def eval(self):
        """Evaluate the model. All algorithms should fit this evaluation pipeline."""
        for agent in self.agents:
            agent.prep_rollout()

        self.logger.eval_init(self.n_rollout_threads)  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.envs.reset()

        eval_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.agents[agent_id].perform(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.envs.step(eval_actions)

            eval_data = (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions)
            self.logger.eval_per_step(eval_data)  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = 0

            eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = 0

            for eval_i in range(self.n_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)  # logger callback when an episode is done

            if eval_episode >= self.algo_args["train"]["perturb_episodes"]:
                self.logger.eval_log(eval_episode)  # logger callback at the end of evaluation
                break

    def eval_adv(self):
        """Evaluate the adversarial attacks. All algorithms should fit this evaluation pipeline."""
        for agent in self.agents:
            agent.prep_rollout()

        self.logger.eval_init(self.n_rollout_threads)  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.envs.reset()

        eval_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_obs_adv = self.attacks[agent_id].perturb(
                    self.agents[agent_id],
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None else None,
                )

                eval_actions, temp_rnn_state = self.agents[agent_id].perform(
                    eval_obs_adv,
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.envs.step(eval_actions)

            eval_data = (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions)
            self.logger.eval_per_step(eval_data)  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = 0

            eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = 0

            for eval_i in range(self.n_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)  # logger callback when an episode is done

            if eval_episode >= self.algo_args["train"]["perturb_episodes"]:
                self.logger.eval_log_adv(eval_episode)  # logger callback at the end of evaluation
                break

    def render(self):
        """Render the model"""
        print("start rendering")

        eval_rnn_states = np.zeros((self.env_num, self.num_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.env_num, self.num_agents, 1), dtype=np.float32)

        for _ in range(self.algo_args['render']['render_episodes']):
            eval_obs, _, eval_available_actions = self.envs.reset()
            eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
            if eval_available_actions is not None:
                eval_available_actions = np.expand_dims(np.array(eval_available_actions), axis=0)
            rewards = 0
            while True:
                eval_actions_collector = []
                for agent_id in range(self.num_agents):
                    eval_obs_adv = self.attacks[agent_id].perturb(
                        self.agents[agent_id],
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions[0] is not None else None,
                    )
                    
                    eval_actions, temp_rnn_state = self.agents[agent_id].perform(
                        eval_obs_adv,
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions is not None else None,
                        deterministic=True,
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))
                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

                eval_obs, _, eval_rewards, eval_dones, _, eval_available_actions = self.envs.step(eval_actions[0])
                rewards += eval_rewards[0][0]
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                if self.manual_render:
                    self.envs.render()
                if self.manual_delay:
                    time.sleep(0.1)
                if eval_dones[0]:
                    print(f'total reward of this episode: {rewards}')
                    break
                
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def restore(self):
        """Restore the model"""
        if self.share_param:
            self.agents[0].restore(self.algo_args['victim']['model_dir'])
        else:
            for agent_id in range(self.num_agents):
                self.agents[agent_id].restore(os.path.join(self.algo_args['victim']['model_dir'], str(agent_id)))

    def save(self):
        """Save the model"""
        if self.share_param:
            self.agents[0].save(self.algo_args['victim']['model_dir'])
        else:
            for agent_id in range(self.num_agents):
                self.agents[agent_id].save(os.path.join(self.algo_args['victim']['model_dir'], str(agent_id)))

    def close(self):
        """Close environment."""
        if self.algo_args['render']['use_render']:
            self.envs.close()
        else:
            self.envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()