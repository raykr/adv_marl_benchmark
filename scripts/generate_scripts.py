import os

ATTACK_CONF = {
    "random_noise": "--run perturbation --algo.num_env_steps 0 --algo.perturb_iters 0 --algo.adaptive_alpha False --algo.targeted_attack False",
    "iterative_perturbation": "--run perturbation --algo.num_env_steps 0  --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack False",
    "random_policy": "--run traitor --algo.num_env_steps 0",
    "traitor": "--run traitor --algo.num_env_steps 5000000",
    "adaptive_action": "--run perturbation --algo.num_env_steps 5000000 --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True",
}

# 如果是list，则遍历所有的值生成命令
# 如果是dict，则属于组合参数，需要特殊处理
TRICKS = {
    "mappo": {
        "entropy_coef": [0.0001, 0.001, 0.1, 0.5, 1.0],
        "gamma": [0.95, 1],
        "hidden_sizes": [[64, 64], [512, 512]],
        "activation_func": ["leaky_relu", "selu", "sigmoid", "tanh"],
        "initialization_method": ["xavier_uniform_"],
        "use_recurrent_policy": [True],
        "use_feature_normalization": [False],
        "lr": [0.00005, 0.005],
        "critic_lr": [0.00005, 0.005],
        "use_gae": [False],
        "use_popart": [False],
        "share_param": [False]
    },
    "maddpg": {
        "expl_noise": [0.001, 0.01, 0.5, 1.0],
        "gamma": [0.95, 1],
        "hidden_sizes": [[64, 64], [512, 512]],
        "activation_func": ["leaky_relu", "selu", "sigmoid", "tanh"],
        "initialization_method": ["xavier_uniform_"],
        "use_recurrent_policy": [True],
        "use_feature_normalization": [False],
        "lr": [0.00005, 0.005],
        "critic_lr": [0.00005, 0.005],
        "n_step": [10, 50],
        "share_param": [False],
        "batch_size": [500, 5000]
    },
    "qmix": {
        "epsilon_anneal_time": [50000, 200000],
        "epsilon_finish": [0.01, 0.1],
        "eps_delta_l": {"epsilon_anneal_time": 80000, "epsilon_finish": 0.24},
        "eps_delta_r": {"epsilon_anneal_time": 104211, "epsilon_finish": 0.01},
        "gamma": [0.95, 1],
        "hidden_sizes": [[64, 64], [256, 256]],
        "activation_func": ["leaky_relu", "selu", "sigmoid", "tanh"],
        "initialization_method": ["xavier_uniform_"],
        "use_recurrent_policy": [True],
        "use_feature_normalization": [False],
        "lr": [0.00005, 0.005],
        "critic_lr": [0.00005, 0.005],
        "n_step": [10, 50],
        "share_param": [False],
        "batch_size": [500, 5000]
    }
}

def generate_train_scripts(env, scenario, algo):
     # 构建数据输出目录，如果没有则创建
    outs_dir = os.path.join("logs", env, scenario, algo)
    # settings目录
    settings_dir = os.path.join("settings", env, scenario)
    # baseline配置文件
    config_path = os.path.join(settings_dir, algo + ".json")
    # 生成脚本文件
    with open(f"train_{env}_{scenario}_{algo}.sh", "w") as f:
        # 生成默认命令
        default_dir = os.path.join(outs_dir, "default")
        os.makedirs(default_dir, exist_ok=True)
        command = f"python -u ../single_train.py --load_config {config_path} --exp_name default > {default_dir}/train.log 2>&1"
        f.write(command + "\n")
        # 遍历tricks
        for key, value in TRICKS[algo].items():
            trick_str = ""
            exp_name = ""
            if isinstance(value, dict):  # 如果是dict，则特殊处理，组合命令
                exp_name = f"{key}"
                # 遍历字典，生成命令
                for k, v in value.items():
                    trick_str += f" --algo.{k} {v}"
                # 生成命令
                log_dir = os.path.join(outs_dir, exp_name)
                os.makedirs(log_dir, exist_ok=True)
                command = f"python -u ../single_train.py --load_config {config_path} --exp_name {exp_name} {trick_str} > {log_dir}/train.log 2>&1"
                f.write(command + "\n")
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
                    log_dir = os.path.join(outs_dir, exp_name)
                    os.makedirs(log_dir, exist_ok=True)
                    command = f"python -u ../single_train.py --load_config {config_path} --exp_name {exp_name} {trick_str} > {log_dir}/train.log 2>&1"
                    f.write(command + "\n")
        
    print("Generate train scripts done!", f"train_{env}_{scenario}_{algo}.sh")


def generate_eval_scripts(env, scenario, algo, slice=False, group="model"):
    models_dir = os.path.join("results", env, scenario, "single", algo)
    attacks = ATTACK_CONF
    # 根据环境不同，生成各自的命令
    # 注意此处algo mappo是用作攻击的，此处设置为固定使用mappo作为攻击RL算法
    base_cfg = f"--algo mappo --env {env}"
    if env == "smac":
        base_cfg += f" --env.map_name {scenario}"
    else:
        base_cfg += f" --env.scenario {scenario}"
        if env == "mamujoco":
            base_cfg += f" --env.agent_conf 3x1"

    # 生成脚本文件
    with open(f"eval_{env}_{scenario}_{algo}.sh", "w") as f:
        # 读取victims_dir目录列表
        victims_dirs = os.listdir(models_dir)
        # 排序,但是default要放在最前
        victims_dirs.remove("default")
        victims_dirs.sort()
        victims_dirs.insert(0, "default")
       
        victim_tuples = []
         # 遍历victims_dirs
        for victim_dir in victims_dirs:
            # 查看victim_dir是否是目录，如果是看子目录有几个
            if os.path.isdir(os.path.join(models_dir, victim_dir)):
                # 读取子目录列表
                sub_victims_dirs = os.listdir(os.path.join(models_dir, victim_dir))
                # 遍历子目录列表
                for sub_victim_dir in sub_victims_dirs:
                    victim_tuples.append((victim_dir, sub_victim_dir))
        
        if group == "model":
            for victim_dir, sub_victim_dir in victim_tuples:
                # 构建数据输出目录，如果没有则创建
                outs_dir = os.path.join(
                    "logs",
                    env,
                    scenario,
                    algo,
                    victim_dir,
                )
                os.makedirs(outs_dir, exist_ok=True)
                # 遍历所有攻击算法
                for attack_method, attack_cfg in attacks.items():
                    # 生成命令
                    command = f"python -u ../single_train.py {base_cfg} --algo.slice {slice} --load_victim {os.path.join(models_dir, victim_dir, sub_victim_dir)} --exp_name {victim_dir}_{attack_method} {attack_cfg} > {outs_dir}/{attack_method}.log 2>&1"
                    f.write(command + "\n")
            f.write("\n")
        elif group == "attack":
            for attack_method, attack_cfg in attacks.items():
                for victim_dir, sub_victim_dir in victim_tuples:
                    # 构建数据输出目录，如果没有则创建
                    outs_dir = os.path.join(
                        "logs",
                        env,
                        scenario,
                        algo,
                        victim_dir,
                    )
                    os.makedirs(outs_dir, exist_ok=True)
                    # 生成命令
                    command = f"python -u ../single_train.py {base_cfg} --algo.slice {slice} --load_victim {os.path.join(models_dir, victim_dir, sub_victim_dir)} --exp_name {attack_method}_{victim_dir} {attack_cfg} > {outs_dir}/{attack_method}.log 2>&1"
                    f.write(command + "\n")
                f.write("\n")
            f.write("\n")
            
    print("Generate eval scripts done!", f"eval_{env}_{scenario}_{algo}.sh")


# generate_train_scripts("mamujoco", "HalfCheetah-6x1", "mappo")
# generate_train_scripts("mamujoco", "HalfCheetah-6x1", "maddpg")
# generate_train_scripts("mamujoco", "Hopper-3x1", "mappo")
# generate_train_scripts("mamujoco", "Hopper-3x1", "maddpg")
# generate_train_scripts("pettingzoo_mpe", "simple_speaker_listener_v4", "mappo")
# generate_train_scripts("pettingzoo_mpe", "simple_speaker_listener_v4", "maddpg")
# generate_train_scripts("pettingzoo_mpe", "simple_spread_v3", "mappo")
# generate_train_scripts("pettingzoo_mpe", "simple_spread_v3", "maddpg")
# generate_train_scripts("smac", "3m", "mappo")
# generate_train_scripts("smac", "3m", "qmix")
# generate_train_scripts("smac", "2s3z", "mappo")
# generate_train_scripts("smac", "2s3z", "qmix")


# generate_eval_scripts("smac", "2s3z", "mappo", group="attack")
# generate_eval_scripts("smac", "2s3z", "qmix", group="attack")