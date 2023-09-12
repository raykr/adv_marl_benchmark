import json
import configparser
import os
from rlplotter.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tblog_to_csv(
    log_path, ytag="test/eval_average_episode_reward_run", exp_name="mappo_iclr", env_name="halfcheetah_6x1"
):
    event_data = EventAccumulator(log_path)
    event_data.Reload()
    reward = event_data.scalars.Items(ytag)
    logger = Logger(env_name, exp_name=exp_name)
    for elm in reward:
        logger.update(score=[elm.value], total_steps=elm.step)


# for root in ["manyagent_swimmer_4x2","HalfCheetah-v2_6x1",]: # "manyagent_swimmer_6x2", "HalfCheetah-v2_2x3",  "Walker2d-v2_2x3", "Walker2d-v2_6x1", "Ant-v2_4x2"]:
#     exps_dir = os.listdir(root)
#     for exp in exps_dir:
#         exps_dir2 = os.listdir(os.path.join(root, exp))
#         for exp2 in exps_dir2:
#             exp_name = exp.split("_")[1] + exp2
#             seeds_dir = os.listdir(os.path.join(root, exp, exp2))
#             for seed in seeds_dir:
#                 log_dir = os.listdir(os.path.join(root, exp, exp2, seed))
#                 log_dir.sort()
#                 log_dir2 = os.listdir(os.path.join(root, exp, exp2, seed, log_dir[-1], "logs"))
#                 log_dir2.sort()
#                 try:
#                     log_path = os.path.join(root, exp, exp2, seed, log_dir[-1], "logs", log_dir2[0])
#                 except:
#                     print(root, exp, exp2, seed)

#                 tblog_to_csv(log_path, exp_name=exp_name, env_name=root.replace("_", ""))


nni_root = "/root/nni-experiments"


config = configparser.ConfigParser()


def covert2csv(nni_root, exp_ids, param):
    for root in exp_ids:
        exps_dir = os.listdir(os.path.join(nni_root, root, "trials"))
        for exp in exps_dir:
            nni_log_dir = os.path.join(nni_root, root, "trials", exp)
            with open(os.path.join(nni_log_dir, "parameter.cfg")) as f:
                config = json.load(f)
            config = config["parameters"]
            algo = config["main_args.algo"]
            env = config["main_args.env"]
            # run = config["main_args.run"]
            run = "traitor"
            map = config["env_args.map_name"]
            seed = config["algo_args.train.seed"]
            v = config[param]
            p = param.split(".")[-1]

            exp_name = f"{p}{v}/seed{seed}"
            env_name = f"./results/{root}/{map}/{algo}/{run}"
            tb_log_dir = os.path.join(nni_root, root, "trials", exp, "tensorboard", "logs")
            print(tb_log_dir, exp_name, env_name)

            tblog_to_csv(tb_log_dir, ytag="env/eval_return_mean", exp_name=exp_name, env_name=env_name)


covert2csv("/root/nni-experiments", ["18eg5nur"], "algo_args.train.lr")
