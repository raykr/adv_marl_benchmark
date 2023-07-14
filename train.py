import os
import argparse
import json
from amb.utils.config_utils import get_defaults_yaml_args, update_args

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
            "igs",
            "qmix",
            "vdn",
            "iql",
            "qtran",
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
            "traitor"
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
        ],
        help="Victim algorithm name. Choose from: maddpg, mappo.",
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
        args["exp_name"] = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
        update_args(unparsed_dict, algo=algo_args, env=env_args)  # update args from command line
    else:  # load config from corresponding yaml file
        if args["load_victim"] != "":
            with open(os.path.join(args["load_victim"], "config.json"), encoding='utf-8') as file:
                victim_config = json.load(file)
            args["victim"] = victim_config["main_args"]["algo"]
            args["env"] = victim_config["main_args"]["env"]
            victim_config["algo_args"]["train"]["model_dir"] = os.path.join(args["load_victim"], "models")

            victim_args = {}
            # "flatten" the victim args
            def update_dict(dict1, dict2):
                for k in dict2:
                    if type(dict2[k]) is dict:
                        update_dict(dict1, dict2[k])
                    else:
                        dict1[k] = dict2[k]
            update_dict(victim_args, victim_config["algo_args"])
            env_args = victim_config["env_args"]
            if args["run"] == "perturbation":
                algo_args, _, _ = get_defaults_yaml_args(args["algo"], args["env"], args["victim"])
            elif args["run"] == "traitor":
                algo_args, _, _ = get_defaults_yaml_args(args["algo"] + "_traitor", args["env"], args["victim"])
            update_args(unparsed_dict, algo=algo_args, env=env_args, victim=victim_args)  # update args from command line
            algo_args = {**algo_args, "victim": victim_args}
        elif args["run"] == "perturbation":
            algo_args, env_args, victim_args = get_defaults_yaml_args(args["algo"], args["env"], args["victim"])
            update_args(unparsed_dict, algo=algo_args, env=env_args, victim=victim_args)  # update args from command line
            algo_args = {**algo_args, "victim": victim_args}
        elif args["run"] == "traitor":
            algo_args, env_args, victim_args = get_defaults_yaml_args(args["algo"] + "_traitor", args["env"], args["victim"])
            update_args(unparsed_dict, algo=algo_args, env=env_args, victim=victim_args)  # update args from command line
            algo_args = {**algo_args, "victim": victim_args}
        else:
            algo_args, env_args, _ = get_defaults_yaml_args(args["algo"], args["env"])
            update_args(unparsed_dict, algo=algo_args, env=env_args)

    # start training
    from amb.runners import get_runner
    runner = get_runner(args["run"], args["algo"])(args, algo_args, env_args)
    if algo_args['render']['use_render']:  # render, not train
        runner.render()
    else:
        runner.run()
    runner.close()


if __name__ == "__main__":
    main()
