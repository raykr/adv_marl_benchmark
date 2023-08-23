import os
import argparse
import json
from amb.utils.config_utils import get_one_yaml_args, update_args

def main():
    """Main function."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--angel",
        type=str,
        default="mappo",
        choices=[
            "maddpg",
            "mappo",
            "igs",
            "qmix",
        ],
        help="Algorithm name. Choose from: maddpg, mappo, igs.",
    )
    parser.add_argument(
        "--demon",
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
        "--run",
        type=str,
        default="dual",
        choices=[
            "dual",
            "perturbation",
            "traitor",
        ],
        help="Runner pipeline name. Choose from: dual, perturbation, traitor.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="smac_dual",
        choices=[
            "smac_dual",
        ],
        help="Environment name. Choose from: smac_dual.",
    )
    parser.add_argument("--exp_name", type=str, default="installtest", help="Experiment name.")
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--load_angel",
        type=str,
        default="",
        help="If set, load existing angel config file and checkpoint file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--load_demon",
        type=str,
        default="",
        help="If set, load existing demon config file and checkpoint file instead of reading from yaml config file.",
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
        args["angel"] = all_config["main_args"]["angel"]
        args["demon"] = all_config["main_args"]["demon"]
        args["run"] = all_config["main_args"]["run"]
        args["env"] = all_config["main_args"]["env"]
        args["exp_name"] = all_config["main_args"]["exp_name"]

        angel_args = all_config["algo_args"]["angel"]
        demon_args = all_config["algo_args"]["demon"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        if args["load_angel"] != "":
            with open(os.path.join(args["load_angel"], "config.json"), encoding='utf-8') as file:
                angel_config = json.load(file)

            if "algo" in angel_config["main_args"]:
                args["angel"] = angel_config["main_args"]["algo"]
                angel_config["algo_args"]["train"]["model_dir"] = os.path.join(args["load_angel"], "models")
                angel_args = angel_config["algo_args"]["train"]
            else:
                args["angel"] = angel_config["main_args"]["angel"]
                angel_config["algo_args"]["angel"]["model_dir"] = os.path.join(args["load_angel"], "models", "angel")
                angel_args = angel_config["algo_args"]["angel"]

            args["env"] = angel_config["main_args"]["env"]
            env_args = angel_config["env_args"]
        else:
            angel_args = get_one_yaml_args(args["angel"])

        if args["load_demon"] != "":
            with open(os.path.join(args["load_demon"], "config.json"), encoding='utf-8') as file:
                demon_config = json.load(file)
            if "algo" in demon_config["main_args"]:
                args["demon"] = demon_config["main_args"]["algo"]
                demon_config["algo_args"]["train"]["model_dir"] = os.path.join(args["load_demon"], "models")
                demon_args = demon_config["algo_args"]["train"]
            else:
                args["demon"] = demon_config["main_args"]["demon"]
                demon_config["algo_args"]["demon"]["model_dir"] = os.path.join(args["load_demon"], "models", "demon")
                demon_args = demon_config["algo_args"]["demon"]

            # args["env"] = demon_config["main_args"]["env"]
            # env_args = demon_config["env_args"]
        else:
            demon_args = get_one_yaml_args(args["demon"])

        if args["load_angel"] == "" and args["load_demon"] == "":
            env_args = get_one_yaml_args(args["env"], type="env")
            
    update_args(unparsed_dict, angel=angel_args, env=env_args, demon=demon_args)  # update args from command line
    algo_args = {"angel": angel_args, "demon": demon_args}

    # start training
    from amb.runners import get_runner
    runner = get_runner(args["run"], args["angel"])(args, algo_args, env_args)
    if algo_args["angel"]['use_render']:  # render, not train
        runner.render()
    else:
        runner.run()
    runner.close()


if __name__ == "__main__":
    main()
