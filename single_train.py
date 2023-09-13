import os
import argparse
import json
from pprint import pprint
from amb.utils.config_utils import get_one_yaml_args, update_args, parse_timestep

def main():
    """Main function."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=[
            "maddpg",
            "mappo",
            "qmix",
        ],
        help="Algorithm name. Choose from: maddpg, mappo, igs.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="single",
        choices=[
            "single",
            "perturbation",
            "traitor",
        ],
        help="Runner pipeline name. Choose from: single, perturbation, traitor.",
    )
    parser.add_argument(
        "--victim",
        type=str,
        default="mappo",
        choices=[
            "maddpg",
            "mappo",
            "qmix"
        ],
        help="Victim algorithm name. Choose from: maddpg, mappo, qmix.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "smacv2",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, smacv2.",
    )
    parser.add_argument("--exp_name", type=str, default="installtest", help="Experiment name.")
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--load_victim",
        type=str,
        default="",
        help="If set, load existing victim config file and checkpoint file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding='utf-8') as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        args["run"] = all_config["main_args"]["run"]
        args["exp_name"] = all_config["main_args"]["exp_name"]

        algo_args = all_config["algo_args"]["train"]
        victim_args = all_config["algo_args"]["victim"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        if args["run"] == "single":
            algo_args = get_one_yaml_args(args["algo"])
        elif args["run"] == "perturbation" or args["run"] == "traitor":
            algo_args = get_one_yaml_args(args["algo"] + "_traitor")

        if args["load_victim"] != "":
            with open(os.path.join(args["load_victim"], "config.json"), encoding='utf-8') as file:
                victim_config = json.load(file)
            args["victim"] = victim_config["main_args"]["algo"]
            args["env"] = victim_config["main_args"]["env"]
            victim_config["algo_args"]["train"]["model_dir"] = os.path.join(args["load_victim"], "models")

            victim_args = victim_config["algo_args"]["train"]
            env_args = victim_config["env_args"]
        else:
            victim_args = {}
            if args["run"] == "perturbation" or args["run"] == "traitor":
                victim_args = get_one_yaml_args(args["victim"])
            env_args = get_one_yaml_args(args["env"], type="env")
            
    update_args(unparsed_dict, algo=algo_args, env=env_args, victim=victim_args)  # update args from command line
    algo_args = {"train": algo_args, "victim": victim_args}

    if "perturb_timesteps" in algo_args["train"]:
        algo_args["train"]["perturb_timesteps"] = parse_timestep(algo_args["train"]["perturb_timesteps"], algo_args["train"]["episode_length"])

    pprint([args, algo_args, env_args])

    # start training
    from amb.runners import get_runner
    runner = get_runner(args["run"], args["algo"])(args, algo_args, env_args)
    if algo_args["train"]['use_render']:  # render, not train
        runner.render()
    else:
        runner.run()
    runner.close()


if __name__ == "__main__":
    main()
