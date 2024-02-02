import os
from scipy.stats import kendalltau
import pandas as pd
import numpy as np


def get_paths(args):
    # 往下walk三级目录，返回一个(env, scenario, algo)三元组的列表
    envs = []
    for env_name in os.listdir(os.path.join(args.out, "data")):
        if not os.path.isdir(os.path.join(args.out, "data", env_name)):
            continue
        if args.env is not None and args.env != env_name:
            continue

        for scenario_name in os.listdir(os.path.join(args.out, "data", env_name)):
            if not os.path.isdir(os.path.join(args.out, "data", env_name, scenario_name)):
                continue
            if args.scenario is not None and args.scenario != scenario_name:
                continue

            for algo_name in os.listdir(os.path.join(args.out, "data", env_name, scenario_name)):
                if not os.path.isdir(os.path.join(args.out, "data", env_name, scenario_name, algo_name)):
                    continue
                if args.algo is not None and args.algo != algo_name:
                    continue

                envs.append((env_name, scenario_name, algo_name))
    return envs


def cal_kendalltau_correlation(excel_path, args):
    xlsx = pd.ExcelFile(excel_path)

    exp_name, vanilla_reward = [], []
    adv_reward = {name: [] for name in xlsx.sheet_names}
    for i, sheet_name in enumerate(xlsx.sheet_names):
        # 读取每个工作表中的特定列数据
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        if i == 0:
            exp_name = df["exp_name"].tolist()
            exp_name.append(exp_name[0])
            exp_name = exp_name[1:]
            # 对于exp_name的每个元素，去掉timestep_前缀，并将default改为final
            exp_name = [name.split("_")[-1] for name in exp_name]
            exp_name = ["final" if name == "default" else name for name in exp_name]

            vanilla_reward = df["vanilla_reward"].tolist()
            vanilla_reward.append(vanilla_reward[0])
            vanilla_reward = vanilla_reward[1:]

        ar = df["adv_reward"].tolist()
        ar.append(ar[0])
        ar = ar[1:]
        adv_reward[sheet_name] = ar

    # 对于exp_name[:-1]的数据，替换为从10开始的等差数列，差值为10
    exp_name[:-1] = np.arange(10, (len(exp_name) - 1) * 10 + 10, 10)

    # 如果vanilla_reward中的元素都相等，则将第一个元素减去1e-6
    if len(set(vanilla_reward)) == 1:
        vanilla_reward[0] = vanilla_reward[0] - 1e-6

    # 计算Kendall Tau相关系数
    kendall_results = {"env": [], "scenario": [], "algo": [], "attack": [], "tau": [], "p": []}
    for name, values in adv_reward.items():
        coef, p = kendalltau(vanilla_reward, values)
        kendall_results["env"].append(args.env)
        kendall_results["scenario"].append(args.scenario)
        kendall_results["algo"].append(args.algo)
        kendall_results["attack"].append(name)
        kendall_results["tau"].append(coef)
        kendall_results["p"].append(p)

    # 将结果写入到excel文件中
    df = pd.DataFrame(kendall_results)
    
    # 往指定的excel文件中追加数据
    output = os.path.join(args.out, 'data', "es_kendalltau_correlation.xlsx")
    mode = "w" if not os.path.exists(output) else "a"
    with pd.ExcelWriter(output, mode=mode, if_sheet_exists="overlay" if mode =="a" else None, engine="openpyxl") as writer:
        # 读取已有的数据
        if mode == "a":
            df_old = pd.read_excel(output)
            # 合并数据
            df = pd.concat([df_old, df], ignore_index=True)
        # 追加数据，不附盖
        df.to_excel(writer, index=False)


