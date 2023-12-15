import os

ATTACK_ALL = [
    "random_noise",
    "iterative_perturbation",
    "random_policy",
    "adaptive_action",
    "traitor",
]

ATTACK_CONF = {
    "random_noise": "--run perturbation --algo.num_env_steps 0 --algo.use_eval False --algo.perturb_iters 0 --algo.adaptive_alpha False --algo.targeted_attack False",
    "iterative_perturbation": "--run perturbation --algo.num_env_steps 0  --algo.use_eval False --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack False",
    "random_policy": "--run traitor --algo.num_env_steps 0 --algo.use_eval False",
    "adaptive_action": "--run perturbation --algo.num_env_steps 5000000 --algo.use_eval False --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True",
    "traitor": "--run traitor --algo.num_env_steps 5000000 --algo.use_eval False",
}

# 如果是list，则遍历所有的值生成命令
# 如果是dict，则属于组合参数，需要特殊处理
TRICKS = {
    "mappo": {
        "entropy_coef": [0.0001, 0.001, 0.1, 0.5, 1.0],
        "gamma": [0.9, 0.95],
        "hidden_sizes": [[64, 64], [256, 256]],
        "activation_func": ["leaky_relu", "selu", "sigmoid", "tanh"],
        "initialization_method": ["xavier_uniform_"],
        "use_recurrent_policy": [True],
        "use_feature_normalization": [False],
        "lr": [0.00005, 0.005],
        "critic_lr": [0.00005, 0.005],
        "use_gae": [False],
        "use_popart": [False],
        "share_param": [False],
    },
    "maddpg": {
        "expl_noise": [0.001, 0.01, 0.5, 1.0],
        "gamma": [0.9, 0.95],
        "hidden_sizes": [[64, 64], [512, 512]],
        "activation_func": ["leaky_relu", "selu", "sigmoid", "tanh"],
        "initialization_method": ["xavier_uniform_"],
        "use_recurrent_policy": [True],
        "use_feature_normalization": [False],
        "lr": [0.00005, 0.005],
        "critic_lr": [0.00005, 0.005],
        "n_step": [5, 10, 50],
        "share_param": [False],
        "batch_size": [500, 5000],
    },
    "qmix": {
        "epsilon_anneal_time": [50000, 200000],
        "epsilon_finish": [0.01, 0.1],
        "eps_delta_l": {"epsilon_anneal_time": 80000, "epsilon_finish": 0.24},
        "eps_delta_r": {"epsilon_anneal_time": 104211, "epsilon_finish": 0.01},
        "gamma": [0.9, 0.95],
        "hidden_sizes": [[64, 64], [256, 256]],
        "activation_func": ["leaky_relu", "selu", "sigmoid", "tanh"],
        "initialization_method": ["xavier_uniform_"],
        "use_recurrent_policy": [True],
        "use_feature_normalization": [False],
        "lr": [0.00005, 0.005],
        "critic_lr": [0.00005, 0.005],
        "n_step": [5, 10, 50],
        "share_param": [False],
        "batch_size": [500, 5000],
    },
}


def generate_train_scripts(conf, file_name):
    # 将conf中带--的转换为字符串
    base_cfg = " ".join([f"{k} {v}" for k, v in conf.items() if k.startswith("--")])

    # 提取变量
    slice = False if "--algo.slice" not in conf else conf["--algo.slice"]
    env = conf["--env"]
    scenario = (
        conf["--env.scenario"] if "--env.scenario" in conf else conf["--env.map_name"]
    )
    if "--env.agent_conf" in conf:
        scenario += "-" + conf["--env.agent_conf"]
    algo = conf["--algo"]

    tricks = TRICKS[algo]

    # 构建数据输出目录，如果没有则创建
    outs_dir = os.path.join("logs", env, scenario, algo)

    # 生成脚本文件
    with open(file_name, "w") as f:
        # 生成默认命令
        default_dir = os.path.join(outs_dir, "default")
        os.makedirs(default_dir, exist_ok=True)
        command = f"python -u ../single_train.py --run single {base_cfg} --exp_name default >> {default_dir}/train.log 2>&1"
        f.write(command + "\n")
        # 遍历tricks
        for key, value in tricks.items():
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
                command = f"python -u ../single_train.py --run single {base_cfg} --exp_name {exp_name} {trick_str} >> {log_dir}/train.log 2>&1"
                f.write(command + "\n")
            elif isinstance(value, list):  # 如果是list，则遍历list生成命令
                print(key, value)
                for v in value:
                    if isinstance(v, list):
                        # 原样转成字符串
                        v = '"' + str(v) + '"'
                        print(v)
                    exp_name = f"{key}_{v}"
                    trick_str = f" --algo.{key} {v}"
                    # 生成命令
                    log_dir = os.path.join(outs_dir, exp_name)
                    os.makedirs(log_dir, exist_ok=True)
                    command = f"python -u ../single_train.py --run single {base_cfg} --exp_name {exp_name} {trick_str} >> {log_dir}/train.log 2>&1"
                    f.write(command + "\n")

        f.write("\n")

    print("Generate train scripts done!", file_name)


def generate_eval_scripts(conf, file_name):
    # 将conf中带--的转换为字符串
    base_cfg = " ".join([f"{k} {v}" for k, v in conf.items() if k.startswith("--")])
    # 提取变量
    slice = False if "--algo.slice" not in conf else conf["--algo.slice"]
    env = conf["--env"]
    scenario = (
        conf["--env.scenario"] if "--env.scenario" in conf else conf["--env.map_name"]
    )
    if "--env.agent_conf" in conf:
        scenario += "-" + conf["--env.agent_conf"]
    algo = conf["--algo"]
    models_dir = conf["models_dir"]
    attacks = conf["attacks"]

    # 生成脚本文件
    with open(file_name, "w") as f:
        # 读取victims_dir目录列表
        victims_dirs = os.listdir(models_dir)
        # 排序
        victims_dirs.sort()
        print(f"victims_dirs: {victims_dirs}")

        # 遍历victims_dirs
        for victim_dir in victims_dirs:
            print(victim_dir)
            # 查看victim_dir是否是目录，如果是看子目录有几个
            if os.path.isdir(os.path.join(models_dir, victim_dir)):
                # 读取子目录列表
                sub_victims_dirs = os.listdir(os.path.join(models_dir, victim_dir))
                # 遍历子目录列表
                for sub_victim_dir in sub_victims_dirs:
                    if attacks is None:
                        attacks = ATTACK_ALL
                    # 遍历所有攻击算法
                    for attack in attacks:
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
                        command = f"python -u ../single_train.py {base_cfg} --load_victim {os.path.join(models_dir, victim_dir, sub_victim_dir)} --exp_name {victim_dir}_{attack} {ATTACK_CONF[attack]} >> {outs_dir}/{attack}.log 2>&1"
                        f.write(command + "\n")
                    f.write("\n")
        print("Generate eval scripts done!", file_name)


# generate_train_scripts(
#     {
#         "--env": "mamujoco",
#         "--env.scenario": "HalfCheetah",
#         "--env.agent_conf": "2x3",
#         "--algo": "mappo",
#         "--algo.num_env_steps": 5000000,
#         "--algo.slice": True,
#         "--algo.slice_timestep_interval": 1000000,
#     },
#     "train_halfcheetah-2x3_mappo.sh",
# )

generate_train_scripts(
    {
        "--env": "mamujoco",
        "--env.scenario": "HalfCheetah",
        "--env.agent_conf": "6x1",
        "--algo": "mappo",
        "--algo.num_env_steps": 10000000,
        "--algo.slice": True,
        "--algo.slice_timestep_interval": 1000000,
    },
    "train_halfcheetah-6x1_mappo.sh",
)

# generate_train_scripts(
#     {
#         "--env": "smac",
#         "--env.map_name": "3m",
#         "--algo": "mappo",
#         "--algo.num_env_steps": 5000000,
#         "--algo.slice": True,
#         "--algo.slice_timestep_interval": 1000000,
#     },
#     "train_3m_mappo.sh",
# )

# generate_train_scripts(
#     {
#         "--env": "smac",
#         "--env.map_name": "2s3z",
#         "--algo": "mappo",
#         "--algo.num_env_steps": 5000000,
#         "--algo.slice": True,
#         "--algo.slice_timestep_interval": 1000000,
#     },
#     "train_2s3z_mappo.sh",
# )

# generate_train_scripts(
#     {
#         "--env": "smac",
#         "--env.map_name": "3m",
#         "--algo": "qmix",
#         "--algo.num_env_steps": 5000000,
#         "--algo.slice": True,
#         "--algo.slice_timestep_interval": 1000000,
#     },
#     "train_3m_qmix.sh",
# )

# generate_train_scripts(
#     {
#         "--env": "smac",
#         "--env.map_name": "2s3z",
#         "--algo": "qmix",
#         "--algo.num_env_steps": 5000000,
#         "--algo.slice": True,
#         "--algo.slice_timestep_interval": 1000000,
#     },
#     "train_2s3z_qmix.sh",
# )

# ---------------------------eval--------------------------------
generate_eval_scripts(
    {
        "--env": "mamujoco",
        "--env.scenario": "HalfCheetah",
        "--env.agent_conf": "6x1",
        "--algo": "mappo",
        "--algo.slice": True,
        "--algo.slice_timestep_interval": 1000000,
        "models_dir": "results/mamujoco/HalfCheetah-6x1/single/mappo",
        "attacks": [
            "random_noise",
            "iterative_perturbation",
            "random_policy",
            "adaptive_action",
            "traitor",
        ],
    },
    "eval_halfcheetah-6x1_mappo.sh",
)
