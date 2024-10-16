import time
from functools import reduce
import numpy as np
import nni
from amb.envs.base_logger import BaseLogger


class SMACLogger(BaseLogger):

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(SMACLogger, self).__init__(args, algo_args, env_args, num_agents, writter, run_dir)
        self.win_key = "won"
        self.infos = [{} for i in range(self.algo_args["train"]["n_rollout_threads"])]

    def get_task_name(self):
        return self.env_args["map_name"]

    def init(self):
        super().init()
        self.last_battles_game = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.float32
        )
        self.last_battles_won = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.float32
        )

    def per_step(self, data):
        super().per_step(data)
        infos = data["infos"]
        filled = data["filled"]
        if len(filled.shape) > 1:
            filled = filled[:, 0]
        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if filled[i]:
                self.infos[i] = infos[i]

    def episode_log(self, actor_train_infos, critic_train_info, buffers):
        for agent_id in range(len(buffers)):
            actor_train_infos[agent_id]["dead_ratio"] = 1 - buffers[agent_id].data["active_masks"].sum() / (
                len(buffers) * reduce(lambda x, y: x * y, list(buffers[agent_id].data["active_masks"].shape)))
        super().episode_log(actor_train_infos, critic_train_info, buffers)

        battles_won = []
        battles_game = []
        incre_battles_won = []
        incre_battles_game = []

        for i, info in enumerate(self.infos):
            if "battles_won" in info[0].keys():
                battles_won.append(info[0]["battles_won"])
                incre_battles_won.append(
                    info[0]["battles_won"] - self.last_battles_won[i]
                )
            if "battles_game" in info[0].keys():
                battles_game.append(info[0]["battles_game"])
                incre_battles_game.append(
                    info[0]["battles_game"] - self.last_battles_game[i]
                )

        incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(incre_battles_game) > 0 else 0.0
        self.writter.add_scalar("env/incre_win_rate", incre_win_rate, self.timestep)

        self.last_battles_game = battles_game
        self.last_battles_won = battles_won

        print(
            "Increase games {:.4f}, win rate on these games is {:.4f}".format(
                np.sum(incre_battles_game),
                incre_win_rate,
            )
        )

    def eval_init(self, n_eval_rollout_threads):
        super().eval_init(n_eval_rollout_threads)
        self.eval_battles_won = 0

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        if self.eval_infos[tid][0][self.win_key] == True:
            self.eval_battles_won += 1

    def eval_log(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards])
        eval_win_rate = self.eval_battles_won / eval_episode
        eval_env_infos = {
            "eval_return_mean": self.eval_episode_rewards,
            "eval_return_std": [np.std(self.eval_episode_rewards)],
            "eval_win_rate": [eval_win_rate],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        ## nni report
        # nni.report_intermediate_result(eval_avg_rew)
        print(
            "Evaluation win rate is {}, evaluation average episode reward is {}.\n".format(
                eval_win_rate, eval_avg_rew
            )
        )

        if self.args["run"] == "single":
            self.log_file.write(
                ",".join(map(str, [self.timestep, eval_avg_rew, eval_win_rate]))
                + "\n"
            )
            self.log_file.flush()

    def eval_log_adv(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards])
        eval_win_rate = self.eval_battles_won / eval_episode
        eval_env_infos = {
            "eval_adv_return_mean": self.eval_episode_rewards,
            "eval_adv_return_std": [np.std(self.eval_episode_rewards)],
            "eval_adv_win_rate": [eval_win_rate],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print(
            "Evaluation adv win rate is {}, evaluation adv average episode reward is {}.\n".format(
                eval_win_rate, eval_avg_rew
            )
        )
        if self.args["run"] == "perturbation":
            self.adv_file.write(
                ",".join(map(str, [
                    self.algo_args["train"]["perturb_epsilon"], 
                    self.algo_args["train"]["perturb_iters"], 
                    self.algo_args["train"]["adaptive_alpha"], 
                    self.algo_args["train"]["perturb_alpha"], 
                    eval_avg_rew, eval_win_rate])) + "\n"
            )
            self.adv_file.flush()
        elif self.args["run"] == "traitor":
            self.log_file.write(
                ",".join(map(str, [self.timestep, eval_avg_rew, eval_win_rate]))
                + "\n"
            )
            self.log_file.flush()

    def eval_log_slice(self, eval_type, slice_tag):
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        eval_battles_won = self.eval_battles_won / self.algo_args["train"]["eval_episodes"]
        self.result_file.write(",".join(map(str, [self.timestep, eval_type, slice_tag, eval_avg_rew, eval_battles_won])) + "\n")
        self.result_file.flush()
        if slice_tag != "final" and slice_tag != "":
            self.writter.add_scalars("env/slice_return_mean", {eval_type: eval_avg_rew}, slice_tag)
            self.writter.add_scalars("env/slice_win_rate", {eval_type: eval_battles_won}, slice_tag)
