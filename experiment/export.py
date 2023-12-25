# 此方法的功能是从实验日志目录中把实验结果导出到csv文件中
# 用法：python export.py <实验日志目录> <导出csv文件名>
# 例如：python export.py ./results/smac/2s3z ./logs/smac_2s3z_results.csv
# 导出的csv文件中包含的列有：实验时间、实验名称、实验参数、实验结果

import argparse
import json
import os
from numpy import empty

import pandas as pd

ATTACKS = [
    "random_noise",
    "iterative_perturbation",
    "random_policy",
    "traitor",
    "adaptive_action",
]

def export_results(env, scenario, algo, attack, out_dir):
    # 构建数据输出目录，如果没有则创建
    csv_file = os.path.join(out_dir, env, scenario, algo, f"{env}_{scenario}_{algo}_{attack}.csv")
    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))

    # 使用dataframe来存储数据
    df = pd.DataFrame(columns=["env", "scenario", "algo", "attack", "exp_name", "vanilla_reward", "adv_reward", "vanilla_win_rate", "adv_win_rate"])
    # 先导出默认配置的实验结果
    _record_row(df, env, scenario, algo, attack, "default")
        # tricks配置文件
    tricks_path = os.path.join("settings", "tricks.json")
    # 读取tricks.json
    with open(tricks_path, "r") as f:
        TRICKS = json.load(f)
        # 按照ATTACKS和TRICKS的顺序，组织csv文件的列名
        for key, value in TRICKS[algo].items():
            exp_name = ""
            if isinstance(value, dict):  # 如果是dict，则特殊处理，组合命令
                exp_name = f"{key}"
                _record_row(df, env, scenario, algo, attack, exp_name)
            elif isinstance(value, list):  # 如果是list，则遍历list生成命令
                for v in value:
                    if isinstance(v, list):
                        # 原样转成字符串
                        v = '"' + str(v) + '"'
                    v = (
                        str(v)
                        .replace('"', "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace(",", "_")
                        .replace(" ", "")
                    )
                    exp_name = f"{key}_{v}"
                    _record_row(df, env, scenario, algo, attack, exp_name)
    # 计算指标
    _calculate_metrics(df)
    # 保存到csv文件的sheet2中
    df.to_csv(csv_file, index=False)
    # print("Experiments results exported to", csv_file)
    print("\n")

def _record_row(df, env, scenario, algo, attack, exp_name):
    #  如果trail_name中含有["random_noise", "iterative_perturbation", "adaptive_action"]
    #  则从perturbation目录中导出
    #  否则从traitor目录中导出
    if attack in ["random_noise", "iterative_perturbation", "adaptive_action"]:
        log_dir = os.path.join(
            "results",
            env,
            scenario,
            "perturbation",
            "mappo-" + algo,
            f"{attack}_{exp_name}",
        )
    elif attack in ["random_policy", "traitor"]:
        log_dir = os.path.join(
            "results", env, scenario, "traitor", "mappo-" + algo, f"{attack}_{exp_name}"
        )
    else:
        return

    # 判断log_dir是否存在，如果不存在，则插入一条空数据
    if not os.path.exists(log_dir) or len(os.listdir(log_dir)) == 0:
        # df增加一行
        df.loc[len(df)] = [env, scenario, algo, attack, exp_name, None, None, None, None]
        return
    # 遍历log_dir，导出实验结果
    for date_dir in os.listdir(log_dir):
        # 读取date_dir下的result.txt
        with open(os.path.join(log_dir, date_dir, "result.txt"), "r") as f:
            # df增加一行
            df.loc[len(df)] = [env, scenario, algo, attack, exp_name, None, None, None, None]
            for line in f.readlines():
                # 如果为空行，则跳过
                if line == "\n":
                    continue
                arr = line.replace("\n", "").split(",")
                if arr[2] == "final":
                    df.loc[len(df) -1, arr[1] + "_reward"] = float(arr[3])
                    if len(arr) == 5:
                        df.loc[len(df) -1, arr[1] + "_win_rate"] = float(arr[4])
                        
def _calculate_metrics(df):
    # 先抓出default行的数据
    df_default = df[df["exp_name"] == "default"]
    baseline_r = df_default["vanilla_reward"].values[0] 
    baseline_ra = df_default["adv_reward"].values[0]
    baseline_w = df_default["vanilla_win_rate"].values[0]

    df["srr"] = (df["adv_reward"] - df["vanilla_reward"]) / df["vanilla_reward"] 
    df["tpr"] = (df["vanilla_reward"] - baseline_r) / baseline_r
    df["trr"] = (df["adv_reward"] - baseline_ra) / baseline_r

    df["w-srr"] = df["adv_win_rate"] - df["vanilla_win_rate"]
    df["w-tpr"] = df["vanilla_win_rate"] - baseline_w
    df["w-trr"] = df["adv_win_rate"] - baseline_w

    print(df)


def combine_exported_csv(env, scenario, algo, out_dir):
    dir_path = os.path.join(out_dir, env, scenario, algo)
    # 先将多个csv文件合并成一个，每个csv文件对应一个sheet
    excel_path = os.path.join(dir_path, f"{env}_{scenario}_{algo}.xlsx")
    file_names = [file for file in os.listdir(dir_path) if file.endswith(".csv")]
    # file_names按照 random_noise， iterative_perturbation， adaptive_action， random_policy， traitor 排序
    file_names.sort(key=lambda x: ATTACKS.index("_".join(os.path.basename(x).split(".")[0].split("_")[3:])))
    # 使用xlsxwriter引擎，可以写入多个sheet
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for file_name in file_names:
            attack = "_".join(os.path.basename(file_name.split(".")[0]).split("_")[3:])
            # 读取CSV文件
            df = pd.read_csv(os.path.join(dir_path, file_name), header=0, index_col=0)
            # 将DataFrame写入不同的sheet
            df.to_excel(writer, sheet_name=f'{attack}', index=True)
            # 删除原来的csv文件
            os.remove(os.path.join(dir_path, file_name))
    return excel_path


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-e", "--env", type=str, default="smac", help="env name")
    args.add_argument(
        "-s", "--scenario", type=str, default="2s3z", help="scenario or map name"
    )
    args.add_argument("-a", "--algo", type=str, default="mappo", help="algo name")
    args.add_argument(
        "-m",
        "--method",
        type=str,
        default="all",
        choices=["random_noise", "iterative_perturbation", "adaptive_action", "random_policy", "traitor", "all"],
        help="attack method",
    )
    args.add_argument("-o", "--out", type=str, default="./data", help="out dir")
    args = args.parse_args()

    if args.method == "all":
        for method in ATTACKS:
            export_results(args.env, args.scenario, args.algo, method, args.out)
    else:
        export_results(args.env, args.scenario, args.algo, args.method, args.out)

    # combine all csv files
    excel_path = combine_exported_csv(args.env, args.scenario, args.algo, args.out)
    print("Experiments results exported to", excel_path)

