import argparse
import json
import os

ATTACK_CONF = {
    "random_noise": "--run perturbation --algo.num_env_steps 0 --algo.perturb_iters 0 --algo.adaptive_alpha False --algo.targeted_attack False",
    "iterative_perturbation": "--run perturbation --algo.num_env_steps 0  --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack False",
    "random_policy": "--run traitor --algo.num_env_steps 0",
    "traitor": "--run traitor --algo.num_env_steps 5000000",
    "adaptive_action": "--run perturbation --algo.num_env_steps 5000000 --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True",
}

ATTACK_CONF_STAGE_2 = {
    "traitor": "--run traitor --algo.num_env_steps 0",
    "adaptive_action": "--run perturbation --algo.num_env_steps 0 --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True",
}


def generate_train_scripts(env, scenario, algo, out_dir, config_path=None, trick=None):
    # 构建数据输出目录，如果没有则创建
    logs_dir = os.path.join(out_dir, "logs", env, scenario, algo)
    # settings目录
    settings_dir = os.path.join("settings", env, scenario)
    # scripts目录
    scripts_dir = os.path.join(out_dir, "scripts")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(settings_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    # baseline配置文件
    config_path = os.path.join(settings_dir, algo + ".json") if config_path is None else config_path
    # tricks配置文件
    tricks_path = os.path.join("settings", "tricks.json")
    # 读取tricks.json
    with open(tricks_path, "r") as f:
        TRICKS = json.load(f)
    # 生成脚本文件
    file_name = os.path.join(scripts_dir, f"train_{env}_{scenario}_{algo}.sh")
    with open(file_name, "w") as f:
        # 生成默认命令
        _write_train_command(f, config_path, "default", logs_dir, trick=trick)
        # 遍历tricks
        os.path.join(settings_dir, "tricks.json")
        for key, value in TRICKS[algo].items():
            trick_str = ""
            exp_name = ""
            if isinstance(value, dict):  # 如果是dict，则特殊处理，组合命令
                exp_name = f"{key}"
                # 遍历字典，生成命令
                for k, v in value.items():
                    trick_str += f" --algo.{k} {v}"
                # 生成命令
                _write_train_command(f, config_path, exp_name, logs_dir, trick_str=trick_str, trick=trick)
            elif isinstance(value, list):  # 如果是list，则遍历list生成命令
                for v in value:
                    if isinstance(v, list):
                        # 原样转成字符串
                        v = '"' + str(v) + '"'
                    trick_str = f" --algo.{key} {v}"
                    v = (
                        str(v)
                        .replace('"', "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace(",", "_")
                        .replace(" ", "")
                    )
                    exp_name = f"{key}_{v}"
                    # 生成命令
                    _write_train_command(f, config_path, exp_name, logs_dir, trick_str=trick_str, trick=trick)

    print(f"python parallel.py -s {file_name} -o {out_dir} -n 2")


def _write_train_command(file, config_path, exp_name, logs_dir, trick_str="", trick=None):
    if trick is not None and trick != exp_name:
        return
    ld = os.path.join(logs_dir, exp_name)
    os.makedirs(ld, exist_ok=True)
    command = f"python -u ../single_train.py --load_config {config_path} --exp_name {exp_name} {trick_str} > {ld}/train.log 2>&1"
    file.write(command + "\n")

def generate_eval_scripts(env, scenario, algo, out_dir, slice=False, stage=0, trick=None, method=None):
    # scripts目录
    scripts_dir = os.path.join(out_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    models_dir = os.path.join("results", env, scenario, "single", algo)
    # 根据环境不同，生成各自的命令
    # 注意此处algo mappo是用作攻击的，此处设置为固定使用mappo作为攻击RL算法
    base_cfg = f"--algo mappo --env {env}"
    if env == "smac":
        base_cfg += f" --env.map_name {scenario}"
    elif env == "mamujoco":
        ss = scenario.split("-")
        base_cfg += f" --env.scenario {ss[0]} --env.agent_conf {ss[1]}"
    elif env == "pettingzoo_mpe":
        ss = scenario.split("-")
        base_cfg += f" --env.scenario {ss[0]} --env.continuous_actions {True if ss[1] == 'continuous' else False}"
    else:
        raise NotImplementedError

    # 生成脚本文件
    file_name = os.path.join(scripts_dir, f"eval_{env}_{scenario}_{algo}.sh" if stage == 0 else f"eval_{env}_{scenario}_{algo}_stage_{stage}.sh")
    with open(file_name, "w") as f:
        # 读取victims_dir目录列表
        victims_dirs = os.listdir(models_dir)
        # 排序,但是default要放在最前
        victims_dirs.remove("default")
        victims_dirs.sort()
        victims_dirs.insert(0, "default")

        victim_tuples = []
        # 遍历victims_dirs
        for victim_dir in victims_dirs:
            # 查看victim_dir是否是目录，如果是，看子目录有几个
            if os.path.isdir(os.path.join(models_dir, victim_dir)):
                # 获取最新的目录
                latest_victim_dir = sorted(os.listdir(os.path.join(models_dir, victim_dir)))[-1]
                victim_tuples.append((victim_dir, latest_victim_dir))
                # # 读取子目录列表
                # sub_victims_dirs = os.listdir(os.path.join(models_dir, victim_dir))
                # # 遍历子目录列表
                # for sub_victim_dir in sub_victims_dirs:
                #     victim_tuples.append((victim_dir, sub_victim_dir))

        for attack_method, attack_cfg in ATTACK_CONF.items() if stage != 2 else ATTACK_CONF_STAGE_2.items():
            if method is not None and method != attack_method:
                continue
            for victim_dir, sub_victim_dir in victim_tuples:
                # 构建数据输出目录，如果没有则创建
                logs_dir = os.path.join(
                    out_dir,
                    "logs",
                    env,
                    scenario,
                    algo,
                    victim_dir,
                )
                os.makedirs(logs_dir, exist_ok=True)

                if stage == 1 and attack_method in ["adaptive_action", "traitor"] and victim_dir != "default":
                    continue
                if stage == 2:
                    if attack_method in ["adaptive_action", "traitor"]:
                        type = "traitor" if attack_method == "traitor" else "perturbation"
                        adv_model_dir = os.path.join("results", env, scenario, type, "mappo-" + algo, f"{attack_method}_default")
                        if not os.path.exists(adv_model_dir):
                            continue
                        latest_models_dir = os.path.join(adv_model_dir, sorted(os.listdir(adv_model_dir))[-1], "models")
                        # 生成命令
                        if trick is not None and trick != victim_dir:
                            continue
                        command = f"python -u ../single_train.py {base_cfg} --algo.slice {slice if victim_dir != 'default' else True} --algo.model_dir {latest_models_dir} --load_victim {os.path.join(models_dir, victim_dir, sub_victim_dir)} --exp_name {attack_method}_{victim_dir} {attack_cfg} > {logs_dir}/{attack_method}.log 2>&1"
                        if victim_dir == "default":
                            command = "# " + command
                        f.write(command + "\n")
                    continue
                # 生成命令
                if trick is not None and trick != victim_dir:
                    continue
                command = f"python -u ../single_train.py {base_cfg} --algo.slice {slice if victim_dir != 'default' else True} --load_victim {os.path.join(models_dir, victim_dir, sub_victim_dir)} --exp_name {attack_method}_{victim_dir} {attack_cfg} > {logs_dir}/{attack_method}.log 2>&1"
                f.write(command + "\n")
            f.write("\n")
        f.write("\n")

    print(f"python parallel.py -s {file_name} -o {out_dir} -n 2")
    if stage == 1:
        print(f"python generate.py eval -e {env} -s {scenario} -a {algo} --stage 2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "eval"],
        help="mode: train or eval",
    )
    parser.add_argument("-e", "--env", type=str, default="smac", help="env name")
    parser.add_argument(
        "-s", "--scenario", type=str, default="2s3z", help="scenario or map name"
    )
    parser.add_argument("-a", "--algo", type=str, default="mappo", help="algo name")
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="stage_0: eval all; stage_one: only eval default model in adaptive_action and traitor; stage_two:load adv model to eval.",
    )
    parser.add_argument("-o", "--out", type=str, default="out", help="out dir")
    parser.add_argument("--slice", action="store_true", help="whether to slice eval")
    parser.add_argument("--config_path", type=str, default=None, help="default config path")
    parser.add_argument("-t", "--trick", type=str, default=None, help="only generate the specified trick scripts")
    parser.add_argument("-m", "--method", type=str, default=None, help="only generate the specified attack algo scripts")
    args, _ = parser.parse_known_args()

    if args.mode == "train":
        generate_train_scripts(args.env, args.scenario, args.algo, args.out, config_path=args.config_path, trick=args.trick)
    elif args.mode == "eval":
        generate_eval_scripts(args.env, args.scenario, args.algo, args.out, slice=args.slice, stage=args.stage, trick=args.trick, method=args.method)
