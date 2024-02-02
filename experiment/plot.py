import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

from utils.plot.boxplot import boxplot_cr
from utils.plot.errorbar import errorbar_metrics
from utils.plot.bar import bar_metrics, barstd_metrics
from utils.plot.tensorboard import plot_train_reward
from utils.plot.necessity import plot_necessity
from utils.plot.line import line_tricks, line_earlystopping

# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macos font
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]  # linux
plt.rc("font", family="Times New Roman")
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号无法显示的问题


# 读取scheme.json
SCHEME_CFG = json.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), "settings/scheme.json")), "r"))


def _read_one_excel_cr(path):
    # 读取Excel文件
    xlsx = pd.ExcelFile(path)
    sheet_names = xlsx.sheet_names

    # Loading data from all sheets
    sheets_data = {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in sheet_names}

    # 取出第一个sheet的exp_name列
    exp_names = sheets_data[sheet_names[0]]["exp_name"]
    # Metrics to calculate mean and std
    origin_metrics = ["TPR", "TRR"]

    # Calculating mean and std for each metric, for each row across all sheets
    row_wise_results = {}
    for i, exp_name in enumerate(exp_names):
        row_metrics = {metric: [] for metric in origin_metrics}

        for sheet in sheet_names:
            for metric in origin_metrics:
                row_metrics[metric].append(sheets_data[sheet].iloc[i][metric])

        # 再把一行的rSRR、TPR、TRR三个指标合并成一个list
        row_metrics["CR"] = [row_metrics[metric] for metric in origin_metrics]
        row_metrics["CR"] = np.array(row_metrics["CR"]).flatten().tolist()

        row_wise_results[exp_name] = {"CR": row_metrics["CR"]}

    return row_wise_results


def _read_one_excel_metrics(path):
    # 读取Excel文件
    xlsx = pd.ExcelFile(path)
    sheet_names = xlsx.sheet_names

    # Loading data from all sheets
    sheets_data = {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in sheet_names}

    # 取出第一个sheet的exp_name列
    exp_names = sheets_data[sheet_names[0]]["exp_name"]
    # Metrics to calculate mean and std
    origin_metrics = ["TPR", "TRR", "rSRR"]

    # Calculating mean and std for each metric, for each row across all sheets
    row_wise_results = {}
    for i, exp_name in enumerate(exp_names):
        row_metrics = {metric: [] for metric in origin_metrics}

        for sheet in sheet_names:
            for metric in origin_metrics:
                row_metrics[metric].append(sheets_data[sheet].iloc[i][metric])

        row_wise_results[exp_name] = {
            metric: {"mean": np.array(row_metrics[metric]).mean(), "std": np.array(row_metrics[metric]).std()}
            for metric in origin_metrics
        }

    return row_wise_results


def plot_tricks(excel_path, argv, ylabel="Episode Reward"):
    # 读取Excel文件
    xlsx = pd.ExcelFile(excel_path)

    # 遍历所有工作表，合并df
    combined_df = pd.DataFrame()
    for i, sheet in enumerate(xlsx.sheet_names):
        # 读取每个工作表中的所有数据
        df = pd.read_excel(excel_path, sheet_name=sheet, header=0, index_col=0)
        # 给df增加一列scheme，用于分组
        if argv["groupby"] == "scheme":
            df["scheme"] = df["exp_name"].apply(lambda x: SCHEME_CFG["tricks"][x][0])
        elif argv["groupby"] == "trick":
            df["scheme"] = df["exp_name"].apply(lambda x: SCHEME_CFG["tricks"][x])
        else:
            raise ValueError(f"Unknown groupby: {argv['groupby']}")

        if i == 0:
            # 填充默认数据列
            combined_df["scheme"] = df["scheme"]
            combined_df["exp_name"] = df["exp_name"]
            if ylabel == "Win Rate":
                combined_df["vanilla_win_rate"] = df["vanilla_win_rate"]
            else:
                combined_df["vanilla_reward"] = df["vanilla_reward"]

        # 对combined_df增加一列，以sheet作为列名，值为df中的adv_reward
        if ylabel == "Win Rate":
            combined_df[sheet] = df["adv_win_rate"]
        else:
            combined_df[sheet] = df["adv_reward"]

    # 把scheme相同的数据分组
    grouped = combined_df.groupby("scheme")
    row_default = grouped.get_group("+").copy()
    for name, group in grouped:
        if name == SCHEME_CFG["tricks"]["default"]:
            continue
        # 合并两个df
        plot_data = pd.concat([row_default, group], ignore_index=True)
        # plot_data去除scheme列
        plot_data = plot_data.drop(columns=["scheme"])
        # 画图
        figurename = os.path.join(
            argv["out"],
            "figures",
            argv["i18n"],
            argv["type"],
            argv["env"],
            argv["scenario"],
            argv["algo"],
            "tricks",
            f'{"winrate" if ylabel == "Win Rate" else "reward"}_{name}.{argv["type"]}',
        )
        line_tricks(plot_data, name, ylabel, figurename, argv)


def plot_metrics(excel_path, argv):
    # 读取Excel文件
    xlsx = pd.ExcelFile(excel_path)

    # 遍历所有工作表，依次画图
    for sheet_name in xlsx.sheet_names:
        # 读取每个工作表中的特定列数据
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        # 画图
        title = f"{argv['env']}_{argv['scenario']}_{argv['algo']}_{sheet_name}"
        figurename = os.path.join(
            argv["out"],
            "figures",
            argv["i18n"],
            argv["type"],
            argv["env"],
            argv["scenario"],
            argv["algo"],
            "metrics",
            f'{title}.{argv["type"]}',
        )
        bar_metrics(df, title, figurename)


def boxplot_envalgo(excel_path, argv):
    row_wise_results = _read_one_excel_cr(excel_path)

    filename = os.path.basename(excel_path).split(".")[0]
    figurename = os.path.join(
        argv["out"], "figures", argv["i18n"], argv["type"], "boxplot", f'{filename}.{argv["type"]}'
    )
    boxplot_cr(row_wise_results, filename, figurename)


def boxplot_algo(argv):
    # 遍历argv["out"]/data下的所有excel，按最后的算法分组
    excel_paths = {"mappo": [], "maddpg": [], "qmix": []}
    for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), argv["out"], "data")):
        for file in files:
            if file.endswith("_tricks.xlsx"):
                # 获取file的文件名
                algo_name = file.split("_")[-2]
                excel_paths[algo_name].append(os.path.join(root, file))

    # 依次遍历每个算法下的所有excel，合并列名相同的数据
    for algo_name, paths in excel_paths.items():
        row_wise_results = {}
        # 遍历这个算法下所有的excel，数据合并到row_wise_results
        for i, path in enumerate(paths):
            one_results = _read_one_excel_cr(path)

            if i == 0:
                row_wise_results = one_results
            else:
                for exp_name, metrics in one_results.items():
                    for metric, values in metrics.items():
                        row_wise_results[exp_name][metric].extend(values)

        figurename = os.path.join(
            argv["out"], "figures", argv["i18n"], argv["type"], "boxplot", f'{algo_name}.{argv["type"]}'
        )
        boxplot_cr(row_wise_results, algo_name, figurename)


def errorbar_mean_attack_metric(excel_path, argv):
    row_wise_results = _read_one_excel_metrics(excel_path)

    title = f"{argv['env']}_{argv['scenario']}_{argv['algo']}"
    figurename = os.path.join(
        argv["out"],
        "figures",
        argv["i18n"],
        argv["type"],
        "errorbar",
        f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}.{argv["type"]}',
    )
    errorbar_metrics(row_wise_results, title, figurename)


def barstd_metric(excel_path, argv):
    row_wise_results = _read_one_excel_metrics(excel_path)
    title = f"{argv['env']}_{argv['scenario']}_{argv['algo']}"
    figurename = os.path.join(
        argv["out"],
        "figures",
        argv["i18n"],
        argv["type"],
        "barstd",
        f'{title}.{argv["type"]}',
    )
    barstd_metrics(row_wise_results, title, figurename)


def plot_early_stopping(env_name, scenario_name, algo_name, argv):
    excel_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        argv["out"],
        "data",
        env_name,
        scenario_name,
        algo_name,
        f"{env_name}_{scenario_name}_{algo_name}_earlystopping.xlsx",
    )

    # 读取Excel文件，遍历每个工作表，依次画图
    xlsx = pd.ExcelFile(excel_path)
    exp_name, vanilla_reward, vanilla_winrate = [], [], []
    adv_reward = {name: [] for name in xlsx.sheet_names}
    adv_winrate = {name: [] for name in xlsx.sheet_names}
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

            vanilla_winrate = df["vanilla_win_rate"].tolist()
            vanilla_winrate.append(vanilla_winrate[0])
            vanilla_winrate = vanilla_winrate[1:]

        ar = df["adv_reward"].tolist()
        ar.append(ar[0])
        ar = ar[1:]
        adv_reward[sheet_name] = ar

        aw = df["adv_win_rate"].tolist()
        aw.append(aw[0])
        aw = aw[1:]
        adv_winrate[sheet_name] = aw

    # 对于exp_name[:-1]的数据，替换为从10开始的等差数列，差值为10
    exp_name[:-1] = np.arange(10, (len(exp_name) - 1) * 10 + 10, 10)

    title = f"{env_name}_{scenario_name}_{algo_name}"
    dirname = os.path.join(
        argv["out"],
        "figures",
        argv["i18n"],
        argv["type"],
        argv["env"],
        argv["scenario"],
        argv["algo"],
        "earlystopping",
    )
    line_earlystopping(
        exp_name, vanilla_reward, adv_reward, title, "Reward", os.path.join(dirname, f"es_reward.{argv['type']}")
    )
    if env_name == "smac":
        line_earlystopping(
            exp_name,
            vanilla_winrate,
            adv_winrate,
            title,
            "Win Rate",
            os.path.join(dirname, f"es_winrate.{argv['type']}"),
        )


def get_paths(args):
    # 往下walk三级目录，返回一个(env, scenario, algo)三元组的列表
    envs = []
    for env_name in os.listdir(os.path.join(args.out, "data")):
        if not os.path.isdir(os.path.join(args.out, "data", env_name)):
            continue
        if args.env is not None and args.env != "None" and args.env != env_name:
            continue

        for scenario_name in os.listdir(os.path.join(args.out, "data", env_name)):
            if not os.path.isdir(os.path.join(args.out, "data", env_name, scenario_name)):
                continue
            if args.scenario is not None and args.scenario != "None" and args.scenario != scenario_name:
                continue

            for algo_name in os.listdir(os.path.join(args.out, "data", env_name, scenario_name)):
                if not os.path.isdir(os.path.join(args.out, "data", env_name, scenario_name, algo_name)):
                    continue
                if args.algo is not None and args.algo != "None" and args.algo != algo_name:
                    continue

                envs.append((env_name, scenario_name, algo_name))
    return envs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="smac", help="env name")
    parser.add_argument("-s", "--scenario", type=str, default="3m", help="scenario or map name")
    parser.add_argument("-a", "--algo", type=str, default="mappo", help="algo name")
    parser.add_argument("-o", "--out", type=str, default="out", help="output dir")
    parser.add_argument("-f", "--file", type=str, default=None, help="Excel file path")
    parser.add_argument("-i", "--i18n", type=str, default="en", choices=["en", "zh"], help="Choose the language")
    parser.add_argument("-t", "--type", type=str, default="png", choices=["png", "pdf"], help="save figure type")
    parser.add_argument(
        "-g", "--groupby", type=str, default="trick", choices=["trick", "scheme"], help="Group by trick or scheme"
    )
    parser.add_argument("--show", action="store_true", help="Whether to show the plot")
    args, _ = parser.parse_known_args()
    argv = vars(args)

    for env_name, scenario_name, algo_name in get_paths(args):
        argv["env"] = env_name
        argv["scenario"] = scenario_name
        argv["algo"] = algo_name

        if argv["file"] is not None:
            excel_path = argv["file"]
        else:
            excel_path = os.path.join(
                argv["out"],
                "data",
                argv["env"],
                argv["scenario"],
                argv["algo"],
                f"{argv['env']}_{argv['scenario']}_{argv['algo']}_tricks.xlsx",
            )

        # 画训练对比曲线图
        plot_train_reward(argv, SCHEME_CFG)

        # 评一个trick方案下所有攻击的reward
        # x轴为trick，y轴为reward，每个trick方案一张图
        plot_tricks(excel_path, argv)
        if argv["env"] == "smac":
            plot_tricks(excel_path, argv, ylabel="Win Rate")

        # 画metrics
        plot_metrics(excel_path, argv)

        # 合并attack、metrics的箱线图，每个（环境+算法）一张图，共12张，看的是算法在特定环境上的不同trick的表现
        boxplot_envalgo(excel_path, argv)

        # 合并attack，画metrics的errorbar图
        errorbar_mean_attack_metric(excel_path, argv)
        # 合并attack，画metrics的bar图，带着std
        barstd_metric(excel_path, argv)

        # 画early stopping图
        plot_early_stopping(env_name, scenario_name, algo_name, argv)

    # 合并env、attack、metrics的箱线图，每个算法一张图，共3张，看的是算法在所有环境上的不同trick的表现
    boxplot_algo(argv)

    # 画metric必要性分析图
    plot_necessity(argv)
