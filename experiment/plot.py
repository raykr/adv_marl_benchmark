import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib.ticker import PercentFormatter

# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macos font
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]  # linux
plt.rc("font", family="Times New Roman")
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号无法显示的问题
# 定义马卡龙配色方案的颜色
macaron_colors_1 = ["#83C5BE", "#FFAAA5", "#FFDDD2", "#FFBCBC", "#F8BBD0", "#FF8C94"]
# 柔和粉红色 (Soft Pink), 天空蓝色 (Sky Blue),淡紫罗兰色 (Lavender),薄荷绿色 (Mint Green),淡黄色 (Light Yellow),杏色 (Apricot),淡橙色 (Light Orange),淡绿色 (Pale Green),淡蓝色 (Pale Blue),淡紫色 (Pale Purple)
macaron_colors_2 = [
    "#F8BBD0",
    "#81D4FA",
    "#B39DDB",
    "#C8E6C9",
    "#FFF59D",
    "#FFCC80",
    "#FFAB91",
    "#C5E1A5",
    "#80DEEA",
    "#CE93D8",
]
# https://blog.csdn.net/slandarer/article/details/114157177
macaron_colors_3 = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2", "#999999"]
# ray_colors = ["#83C5BE", "#FFDDD2", "#FFAB91", "#FFAB40", "#FFE0B2", "#FF8C94", '#cccccc']
# 柔和粉红色 (Soft Pink), 天空蓝色 (Sky Blue),淡紫罗兰色 (Lavender), 淡黄色 (Light Yellow),杏色 (Apricot),淡橙色 (Light Orange),淡绿色 (Pale Green),淡蓝色 (Pale Blue),淡紫色 (Pale Purple)
ray_colors = ["#F8BBD0", "#81D4FA", "#B39DDB", "#FFBCBC", "#FFAB91", "#C5E1A5", "#80DEEA", "#CE93D8"]
# 蓝色, 橙色, 绿色, 红色, 紫色, 棕色, 粉红色, 黄绿色, 青色
sci_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf"]
# 定义不同的点形状和颜色
line_markers = ["o", "s", "D", "^", "v", "<", ">", "p", "h", "x", "*", "+", "1", "2", "3", "4"]
line_colors = [
    "b",
    "r",
    "g",
    "c",
    "m",
    "y",
    "k",
    "orange",
    "pink",
    "brown",
    "gray",
    "purple",
    "olive",
    "cyan",
    "navy",
    "teal",
]
# 彩虹马卡龙
rainbow_colors = [
    "#FA7F6F",  # 红色系
    "#FFAAA5",  # 橙色系
    "#FFF59D",  # 黄色系
    "#C5E1A5",  # 绿色系
    "#8ECFC9",  # 青色系
    "#81D4FA",  # 蓝色系
    "#B39DDB",  # 紫色系
]
macaron_rainbow_colors = [
    # 红色系
    "#ff9999",
    "#ff7f7f",
    "#ff6666",
    "#ff4c4c",
    "#ff3232",
    # 橙色系
    "#ffcc99",
    "#ffb57f",
    "#ff9f66",
    "#ff8a4c",
    "#ff7432",
    # 黄色系
    "#ffff99",
    "#ffff7f",
    "#ffff66",
    "#ffff4c",
    "#ffff32",
    # 绿色系
    "#99ff99",
    "#7fff7f",
    "#66ff66",
    "#4cff4c",
    "#32ff32",
    # 青色系
    "#99ffff",
    "#7fffff",
    "#66ffff",
    "#4cffff",
    "#32ffff",
    # 蓝色系
    "#9999ff",
    "#7f7fff",
    "#6666ff",
    "#4c4cff",
    "#3232ff",
    # 紫色系
    "#ff99ff",
    "#ff7fff",
    "#ff66ff",
    "#ff4cff",
    "#ff32ff",
]
rainbow_colors_2 = [
    "#EEA1AB",
    "#EEA29B",
    "#EEA385",
    "#EEA45C",
    "#E3AB58",
    "#D8B059",
    "#CFB359",
    "#C7B758",
    "#B5BC58",
    "#A9BF58",
    "#98C158",
    "#7FC558",
    "#6AC578",
    "#6AC493",
    "#6BC3AE",
    "#6BC3B7",
    "#6BC2BF",
    "#6CC2C5",
    "#6CC1D5",
    "#6DC0DE",
    "#6DBFE9",
    "#99B9F3",
    "#BCB1F3",
    "#C9ABF3",
    "#D6A5F3",
    "#EC9DDC",
    "#ED9ED0",
    "#EE9FC4",
    "#EEA0B9",
]

i18n = {
    "zh": {
        "exp_name": "实现细节",
        "vanilla_reward": "正常训练",
        "adv_reward": "对抗攻击",
        "vanilla_win_rate": "攻击前胜率",
        "adv_win_rate": "攻击后胜率",
        "srr": "自身鲁棒性",
        "tpr": "Trick性能",
        "trr": "Trick鲁棒性",
        "w-srr": "自身鲁棒性",
        "w-tpr": "Trick性能",
        "w-trr": "Trick鲁棒性",
        "Reward change rate": "回报变化率",
        "adaptive_action": "自适应动作扰动",
        "iterative_perturbation": "最优动作抑制扰动",
        "random_noise": "随机噪声",
        "random_policy": "随机策略",
        "traitor": "零和博弈",
        "reward": "平均回报",
        "trick": "实现细节",
        "A": "探索与利用",
        "B": "网络架构",
        "C": "优化器",
        "D": "优势估计",
        "E": "多智能体特性",
    },
    "en": {
        "exp_name": "Trick",
        "vanilla_reward": "Vanilla",
        "adv_reward": "Adversarial",
        "vanilla_win_rate": "Vanilla Win Rate",
        "adv_win_rate": "Adversarial Win Rate",
        "srr": "Self Robustness Rate",
        "tpr": "Trick Performance Rate",
        "trr": "Trick Robustness Rate",
        "w-srr": "Self Robustness Rate (Win Rate)",
        "w-tpr": "Trick Performance Rate (Win Rate)",
        "w-trr": "Trick Robustness Rate (Win Rate)",
        "Reward change rate": "Reward change rate",
        "adaptive_action": "Adaptive Action",
        "iterative_perturbation": "Iterative Perturbation",
        "random_noise": "Random Noise",
        "random_policy": "Random Policy",
        "traitor": "Traitor",
        "reward": "Episode Reward",
        "trick": "Trick",
        "A": "Exploration and Exploitation",
        "B": "Network Architecture",
        "C": "Optimizer",
        "D": "Advantage Estimation",
        "E": "Multi-Agent Feature",
    },
}

YLIM = {
    "pettingzoo_mpe_simple_speaker_listener_v4-continuous_maddpg": [-110, -10],
    "pettingzoo_mpe_simple_spread_v3-continuous_maddpg": [-120, -20],
}

BOXPLOT_YLIM = {
    "mappo": [-1, 1],
    "maddpg": [-5, 5],
    "qmix": [-1, 1],
}

# 读取scheme.json
SCHEME_CFG = json.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), "settings/scheme.json")), "r"))


def plot_trick_reward(excel_path, argv, ylabel="Episode Reward"):
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
        _plot_line(plot_data, excel_path, "trick", name, ylabel, argv)


def plot_attack_reward(excel_path, argv):
    # 读取Excel文件
    xlsx = pd.ExcelFile(excel_path)

    # 遍历所有工作表，依次画图
    for i, sheet_name in enumerate(xlsx.sheet_names):
        # 读取每个工作表中的特定列数据
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        # 只保留df的exp_name, vanilla_reward, adv_reward列
        df = df[["exp_name", "vanilla_reward", "adv_reward"]]
        # 画图
        _plot_bar(df, excel_path, "attack", sheet_name, argv)


def _plot_bar(df, excel_path, category, name, argv):
    filename = os.path.basename(excel_path).split(".")[0]
    display = i18n[argv["i18n"]]

    # 新画布
    golden_ratio = 1.618
    width = 15  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))

    # 计算列数和条形的宽度
    num_columns = len(df.columns) - 1  # 减去exp_name列
    bar_width = 0.12 if len(df) < 10 else 0.25
    space_between_bars = 0.02 if len(df) < 10 else 0
    total_bar_space = num_columns * bar_width + (num_columns - 1) * space_between_bars

    # 生成每个柱状图的位置
    positions = np.arange(len(df))
    bar_positions = [positions + (bar_width + space_between_bars) * i for i in range(num_columns)]

    # 确保组间有足够的间隔
    for idx, pos in enumerate(bar_positions):
        bar_positions[idx] = pos + (idx // num_columns)

    # 创建柱状图
    for i, column in enumerate(df.columns[1:]):
        plt.bar(
            bar_positions[i], df[column], color=ray_colors[i], width=bar_width, edgecolor="grey", label=display[column]
        )

    # 添加图表细节
    # 判断YLIM是否有该filename的key，如果有，则设置Y轴范围
    if filename in YLIM:
        plt.ylim(YLIM[filename])
    # plt.xlabel(display[scheme[0]])
    plt.xticks(positions + total_bar_space / 2, df["exp_name"], rotation=0 if len(df) < 10 else 45, ha="right")
    plt.ylabel(display["reward"])
    # 在y=0处添加一条水平线
    plt.axhline(y=0, color="grey", linestyle="--", linewidth=1)
    # plt.title(scheme_name)
    plt.legend(ncol=10, frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1))
    plt.tight_layout()

    # 保存图表到文件
    # ./out/figures/en/png/smac/3m/mappo/smac_3m_mappo_A1.png
    save_dir = os.path.join(
        argv["out"], "figures", argv["i18n"], argv["type"], argv["env"], argv["scenario"], argv["algo"], category
    )
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(
        save_dir, f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}_{category}_{name}.{argv["type"]}'
    )
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    print(f"Saved to {figure_name}")

    # 展示图表
    if argv["show"]:
        plt.show()


def plot_metrics(excel_path, argv):
    # 读取Excel文件
    xlsx = pd.ExcelFile(excel_path)

    # 遍历所有工作表，依次画图
    for i, sheet_name in enumerate(xlsx.sheet_names):
        # 读取每个工作表中的特定列数据
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        # 画图
        _plot_metrics(df, excel_path, "metrics", sheet_name, argv)


def _plot_metrics(df, excel_path, category, name, argv):
    filename = os.path.basename(excel_path).split(".")[0]
    display = i18n[argv["i18n"]]

    # 新画布
    golden_ratio = 1.618
    width = 15  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    fig, ax = plt.subplots(figsize=(width, height))

    n_categories = len(df)
    bar_width = 0.25
    r1 = np.arange(n_categories)
    r2 = [x + bar_width for x in r1]
    r3 = [x + 2 * bar_width for x in r1]

    # 绘制 SRR 的折线图
    ax.plot(r3, df["SRR"], color="grey", label="SRR", marker="o", linestyle="--", linewidth=1.5)

    # 绘制其他柱状图
    ax.bar(r1, df["TPR"], color=macaron_colors_1[0], width=bar_width, edgecolor="grey", label="TPR")
    ax.bar(r2, df["TRR"], color=macaron_colors_1[1], width=bar_width, edgecolor="grey", label="TRR")
    ax.bar(r3, df["rSRR"], color=macaron_colors_1[2], width=bar_width, edgecolor="grey", label="rSRR")

    # 在y=0处添加一条水平线
    ax.axhline(y=0, color="black", linewidth=0.5)

    # 设置Y轴为百分比格式
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    # ax.set_xlabel('实现细节')
    ax.set_ylabel("Reward change rate")
    ax.set_title(f"{name}")
    ax.set_xticks([r + bar_width for r in range(n_categories)])
    ax.set_xticklabels(df["exp_name"], rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()

    # 保存图表到文件
    # ./out/figures/en/png/smac/3m/mappo/smac_3m_mappo_A1.png
    save_dir = os.path.join(
        argv["out"], "figures", argv["i18n"], argv["type"], argv["env"], argv["scenario"], argv["algo"], category
    )
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(
        save_dir, f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}_{category}_{name}.{argv["type"]}'
    )
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figure_name}")


def boxplot_mean_attack_metric(excel_path, argv):
    row_wise_results = _read_one_excel_cr(excel_path)
    _boxplot_cr(row_wise_results, os.path.basename(excel_path).split(".")[0], argv)


def boxplot_mean_attack_metric_env(argv):
    # 遍历argv["out"]/data下的所有excel，按最后的算法分组
    excel_paths = {"mappo": [], "maddpg": [], "qmix": []}
    for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), argv["out"], "data")):
        for file in files:
            if file.endswith(".xlsx"):
                algo_name = file.split("_")[-1].split(".")[0]
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

        _boxplot_cr(row_wise_results, algo_name, argv)


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


def _boxplot_cr(row_wise_results, filename, argv):
    # 获取row_wise_results中的所有keys
    exp_names = list(row_wise_results.keys())
    # 设置每个箱线图的位置
    positions = range(1, len(exp_names) + 1)

    # Plotting the line chart with std as the shaded area
    golden_ratio = 1.618
    width = 15  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))

    for exp_name, pos in zip(exp_names, positions):
        plt.boxplot(
            row_wise_results[exp_name]["CR"],
            positions=[pos],
            widths=0.5,
            capwidths=0.4,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            boxprops=dict(facecolor=rainbow_colors_2[pos]),
            meanprops=dict(marker="o", markerfacecolor="black", markeredgecolor="black", markersize=4),
            medianprops=dict(marker=None, color="black", linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor=rainbow_colors_2[pos], markeredgecolor=rainbow_colors_2[pos]),
        )

    algo_name = filename.split("_")[-1]
    if filename in BOXPLOT_YLIM:
        plt.ylim(BOXPLOT_YLIM[algo_name])
    # 在y=0处添加一条水平线
    plt.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    # 设置Y轴为百分比格式
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.grid(True, axis='both', linestyle='--')
    plt.xticks(positions, exp_names, rotation=35, ha="right")
    plt.xlabel("Tricks")
    plt.ylabel("Comprehensive Robustness Change Rate")
    plt.title(filename)
    plt.tight_layout()

    # 保存图表到文件
    category = "boxplot"
    save_dir = os.path.join(argv["out"], "figures", argv["i18n"], argv["type"], category)
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(save_dir, f'{category}_{filename}.{argv["type"]}')
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figure_name}")


def errorbar_mean_attack_metric(excel_path, argv):
    row_wise_results = _read_one_excel_metrics(excel_path)
    _errbar_metrics(row_wise_results, "mean_attack", argv)


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


def _errbar_metrics(row_wise_results, filename, argv):
    # 获取row_wise_results中的所有keys
    exp_names = list(row_wise_results.keys())
    # 位置
    positions = range(len(exp_names))

    # Plotting the line chart with std as the shaded area
    golden_ratio = 1.618
    width = 18  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))

    # 对每个x坐标的点进行轻微的水平偏移
    offset = 0.1  # 偏移量
    num_points_per_x = len(row_wise_results[exp_names[0]].keys())
    for idx, cat in enumerate(positions):
        for point, metric in enumerate(row_wise_results[exp_names[idx]].keys()):
            x_val = cat + (point - num_points_per_x / 2) * offset  # 计算偏移后的x值
            plt.errorbar(
                x_val,
                row_wise_results[exp_names[idx]][metric]["mean"],
                yerr=row_wise_results[exp_names[idx]][metric]["std"],
                fmt=line_markers[point],
                color=sci_colors[point],
                capsize=0,
                ecolor=sci_colors[point],
                alpha=0.4,
            )

    # 将errorbar的点按照metric连接起来，形成折线图
    for point, metric in enumerate(row_wise_results[exp_names[0]].keys()):
        x_vals = [cat + (point - num_points_per_x / 2) * offset for cat in positions]
        y_vals = [row_wise_results[exp_name][metric]["mean"] for exp_name in exp_names]
        plt.plot(x_vals, y_vals, linestyle="--", marker=line_markers[point], color=sci_colors[point], label=metric)

    # 在y=0处添加一条水平线
    plt.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    # 设置Y轴为百分比格式
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.grid(True, axis='both', linestyle='--')
    plt.xticks(positions, exp_names, rotation=35, ha="right")
    plt.xlabel("Tricks")
    plt.ylabel("Robustness Change Rate")
    plt.title(f"{argv['env']}_{argv['scenario']}_{argv['algo']}")
    plt.legend()
    plt.tight_layout()

    # 保存图表到文件
    category = "errorbar"
    save_dir = os.path.join(
        argv["out"], "figures", argv["i18n"], argv["type"], argv["env"], argv["scenario"], argv["algo"], category
    )
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(
        save_dir, f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}_{category}_{filename}.{argv["type"]}'
    )
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figure_name}")


def _plot_line(df, excel_path, category, name, ylabel, argv):
    filename = os.path.basename(excel_path).split(".")[0]
    display = i18n[argv["i18n"]]

    # 新画布
    # golden_ratio = 1.618
    # width = 15  # 假设宽度为10单位
    # height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(12, 8))

    # 绘制每一行的数据
    # 将df列名按照i18n进行替换
    df.columns = [display[col] if col in display else col for col in df.columns]
    for i, row in df.iterrows():
        plt.plot(
            row.index[1:],
            row.values[1:],
            linestyle="--",
            marker=line_markers[i],
            color=line_colors[i],
            label=row[display["exp_name"]],
        )

    # 判断YLIM是否有该filename的key，如果有，则设置Y轴范围
    # numeric_df = df.select_dtypes(include=[np.number])
    # max_value = numeric_df.max().max()
    # min_value = numeric_df.min().min()
    # if filename in YLIM and max_value <= YLIM[filename][1] and min_value >= YLIM[filename][0]:
    #     plt.ylim(YLIM[filename])
    # 设置图表标题和坐标轴标签
    plt.title(f"{argv['env']}_{argv['scenario']}_{argv['algo']}")
    plt.xlabel("Adversarial Methods")
    plt.ylabel(ylabel)

    # 判断YLIM是否有该filename的key，如果有，则设置Y轴范围
    # if filename in YLIM:
    #     plt.ylim(YLIM[filename])

    # 添加网格虚线
    plt.grid(True, linestyle="--")

    # 显示图例
    plt.legend()

    # 保存图表到文件
    # ./out/figures/en/png/smac/3m/mappo/smac_3m_mappo_A1.png
    save_dir = os.path.join(
        argv["out"], "figures", argv["i18n"], argv["type"], argv["env"], argv["scenario"], argv["algo"], category
    )
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(
        save_dir,
        f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}{"_winrate" if ylabel == "Win Rate" else ""}_{category}_{name}.{argv["type"]}',
    )
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figure_name}")

    # 展示图表
    if argv["show"]:
        plt.show()


def plot_train_reward(argv):
    # 根据env, scenario, algo,找到对应的文件夹
    trail_dir = os.path.join("results", argv["env"], argv["scenario"], "single", argv["algo"])

    _plot_tb_data(argv["groupby"], trail_dir, "env/train_episode_rewards", "Episode Reward", 0.9, argv)
    _plot_tb_data(argv["groupby"], trail_dir, "env/incre_win_rate", "Win Rate", 0.9, argv)
    _plot_tb_data(argv["groupby"], trail_dir, "env/eval_return_mean", "Episode Reward", 0.6, argv)
    _plot_tb_data(argv["groupby"], trail_dir, "env/eval_win_rate", "Win Rate", 0.6, argv)


def _plot_tb_data(groupby, trail_dir, tag_name, ylabel, weight, argv):
    skip_tag = False
    dfs = {}

    listdir = os.listdir(trail_dir)
    # 先遍历SCHEME_CFG["tricks"]，再去check目录更好，因为可以保证顺序
    for trick_name, trick_tag in SCHEME_CFG["tricks"].items():
        # 如果目录中没有这一项trick，跳过
        if trick_name not in listdir:
            continue

        # 同时支持scheme和trick两种分组方式，因此需要替换trick_tag
        trick_tag = trick_tag if groupby == "trick" else trick_tag[0]

        if trick_tag not in dfs:
            dfs[trick_tag] = pd.DataFrame(columns=["step", trick_name])

        # 读取TensorBoard日志
        log_dir = os.path.join(trail_dir, trick_name)
        # 取出log_dir下最新的子文件夹
        log_dir = sorted(os.listdir(log_dir))[-1]
        log_dir = os.path.join(trail_dir, trick_name, log_dir, "logs")

        # 判断log_dir是否存在，如果不存在，打印后跳过
        if not os.path.exists(log_dir):
            print(f"{log_dir} not exists")
            continue

        # 创建EventAccumulator对象以读取TensorBoard日志
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        # 判断tag_name是否在event_acc中，如果不在，打印后跳过
        if tag_name not in event_acc.Tags()["scalars"]:
            skip_tag = True
            break
        tag_values = event_acc.Scalars(tag_name)

        # 每个trick_name对应一列
        dfs[trick_tag][trick_name] = np.nan
        for value in tag_values:
            # 判断df中是否有step=value.step的行，如果没有，新增一行
            if not dfs[trick_tag][dfs[trick_tag]["step"] == value.step].empty:
                dfs[trick_tag].loc[dfs[trick_tag]["step"] == value.step, trick_name] = value.value

            else:
                # 新增一行全NaN的数据
                dfs[trick_tag].loc[len(dfs[trick_tag])] = [np.nan for _ in range(len(dfs[trick_tag].columns))]
                # 填充列
                dfs[trick_tag].loc[len(dfs[trick_tag]) - 1, "step"] = value.step
                dfs[trick_tag].loc[len(dfs[trick_tag]) - 1, trick_name] = value.value

    # 画图
    if not skip_tag:
        _plot_train_line(dfs, tag_name.split("/")[1], ylabel, weight, argv)


def _plot_train_line(dfs, tag_name, ylabel, weight, argv):
    # 将tag_data转为df
    for tag, data in dfs.items():
        if tag == SCHEME_CFG["tricks"]["default"]:
            continue

        # Create the plot with shaded area in a similar color but more transparent
        plt.figure(figsize=(9, 6))

        # 按step升序排序
        data = data.sort_values(by="step", ascending=True)
        # 画图
        # 获取data的列名
        column_names = data.columns.values.tolist()
        # 去除step列
        column_names.remove("step")

        # 画默认曲线
        default_df = dfs[SCHEME_CFG["tricks"]["default"]]
        smoothed_values = tensorboard_smoothing(default_df["default"], weight=weight)
        plt.plot(default_df["step"], smoothed_values, color=line_colors[0], label="default")
        plt.fill_between(default_df["step"], smoothed_values, default_df["default"], color=line_colors[0], alpha=0.2)

        # 画trick曲线
        for i, column_name in enumerate(column_names):
            # 从data中挑出step和column_name两列
            clean_data = data[["step", column_name]]
            # 去除data中column_name列为NaN的行
            clean_data = clean_data.dropna(subset=[column_name])
            clean_data = clean_data.reset_index(drop=True)

            # Apply TensorBoard-style smoothing
            smoothed_values = tensorboard_smoothing(clean_data[column_name], weight=weight)
            plt.plot(clean_data["step"], smoothed_values, color=line_colors[i + 1], label=column_name)
            plt.fill_between(
                clean_data["step"], smoothed_values, clean_data[column_name], color=line_colors[i + 1], alpha=0.2
            )

        plt.title(f"{argv['env']}_{argv['scenario']}_{argv['algo']}")
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle="--")
        plt.tight_layout()

        save_dir = os.path.join(
            argv["out"], "figures", argv["i18n"], argv["type"], argv["env"], argv["scenario"], argv["algo"], tag_name
        )
        os.makedirs(save_dir, exist_ok=True)
        figure_name = os.path.join(
            save_dir, f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}_{tag_name}_{tag}.{argv["type"]}'
        )
        plt.savefig(figure_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved to {figure_name}")


def tensorboard_smoothing(values, weight):
    if len(values) == 0:
        return []
    last = values[0]
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def bar_mean_attack_metric(excel_path, argv):
    row_wise_results = _read_one_excel_metrics(excel_path)
    _barstd_metrics(row_wise_results, "mean_attack", argv)


def _barstd_metrics(row_wise_results, filename, argv):
    # 获取row_wise_results中的所有keys
    exp_names = list(row_wise_results.keys())
    # 位置
    positions = range(len(exp_names))

    # Plotting the line chart with std as the shaded area
    golden_ratio = 1.618
    width = 18  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))

    # 对每个x坐标的点进行轻微的水平偏移
    offset = 0.2  # 偏移量
    num_points_per_x = len(row_wise_results[exp_names[0]].keys())
    for idx, cat in enumerate(positions):
        for point, metric in enumerate(row_wise_results[exp_names[idx]].keys()):
            x_val = cat + (point - num_points_per_x / 2) * offset  # 计算偏移后的x值
            if idx == 0:
                plt.bar(
                    x_val,
                    row_wise_results[exp_names[idx]][metric]["mean"],
                    yerr=row_wise_results[exp_names[idx]][metric]["std"],
                    edgecolor="grey",
                    width=0.2,
                    capsize=3,
                    ecolor="#aaa",
                    color=macaron_colors_1[point],
                    label=metric,
                )
            else:
                plt.bar(
                    x_val,
                    row_wise_results[exp_names[idx]][metric]["mean"],
                    yerr=row_wise_results[exp_names[idx]][metric]["std"],
                    edgecolor="grey",
                    width=0.2,
                    capsize=3,
                    ecolor="#aaa",
                    color=macaron_colors_1[point],
                )

    # 在y=0处添加一条水平线
    plt.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    # 设置Y轴为百分比格式
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.grid(True, axis='both', linestyle='--')
    plt.xticks(positions, exp_names, rotation=35, ha="right")
    plt.xlabel("Tricks")
    plt.ylabel("Robustness Change Rate")
    plt.title(f"{argv['env']}_{argv['scenario']}_{argv['algo']}")
    plt.legend()
    plt.tight_layout()

    # 保存图表到文件
    category = "barstd"
    save_dir = os.path.join(
        argv["out"], "figures", argv["i18n"], argv["type"], argv["env"], argv["scenario"], argv["algo"], category
    )
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(
        save_dir, f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}_{category}_{filename}.{argv["type"]}'
    )
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figure_name}")


def plot_necessity(argv):
    selected_trick = "entropy_coef_0.0001"
    # 组织数据
    # 遍历argv["out"]/data下的所有excel，按最后的算法分组
    excel_paths = {"mappo": [], "maddpg": [], "qmix": []}
    for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), argv["out"], "data")):
        for file in files:
            if file.endswith(".xlsx"):
                algo_name = file.split("_")[-1].split(".")[0]
                excel_paths[algo_name].append(os.path.join(root, file))

    # 取mappo算法下的所有excel
    combined_df = pd.DataFrame()
    for path in excel_paths["mappo"]:
        if "mamujoco" in path:
            continue
        # 读取iterative_perturbation sheet的数据
        df = pd.read_excel(path, sheet_name="iterative_perturbation", header=0)
        # 取出exp_name为default和gamma_0.9两行，合并到combined_df
        combined_df = pd.concat([combined_df, df[df["exp_name"].isin(["default", selected_trick])]])
    print(combined_df)

    # 画原始hat图
    draw_origin_hat(combined_df, selected_trick, argv)

    # 画新指标的hat图
    draw_self_metrics_hat(combined_df, selected_trick, argv)
    draw_trick_metrics_hat(combined_df, selected_trick, argv)


def draw_origin_hat(combined_df, trick_name, argv):
    # xlabels为combined_df的scenario列，去重
    xlabels = combined_df["scenario"].unique().tolist()
    # 取exp_name为default的before_reward列
    before_reward = combined_df[combined_df["exp_name"] == "default"]["before_reward"].tolist()
    vanilla_reward = combined_df[combined_df["exp_name"] == "default"]["vanilla_reward"].tolist()
    adv_reward = combined_df[combined_df["exp_name"] == "default"]["adv_reward"].tolist()
    trick_vanilla_reward = combined_df[combined_df["exp_name"] == trick_name]["vanilla_reward"].tolist()
    trick_adv_reward = combined_df[combined_df["exp_name"] == trick_name]["adv_reward"].tolist()

    # Plotting the line chart with std as the shaded area
    golden_ratio = 1.618
    width = 15  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))
    fig, ax = plt.subplots(figsize=(width, height))
    hat_graph(
        ax,
        xlabels,
        [before_reward, vanilla_reward, adv_reward, trick_vanilla_reward, trick_adv_reward],
        [
            "Before Reward",
            "Vanilla Reward",
            "Adversarial Attack Reward",
            f"Vanilla Reward with {trick_name}",
            f"Adversarial Attack Reward with {trick_name}",
        ],
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Environments")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards in different environments of MAPPO")
    ax.set_ylim(-70, 70)
    ax.legend()

    fig.tight_layout()

    # 保存图表到文件
    category = "necessity"
    save_dir = os.path.join(argv["out"], "figures", argv["i18n"], argv["type"], category)
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(save_dir, f'{trick_name}_origin.{argv["type"]}')
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figure_name}")


def draw_self_metrics_hat(combined_df, trick_name, argv):
    # xlabels为combined_df的scenario列，去重
    xlabels = combined_df["scenario"].unique().tolist()
    start = [0, 0, 0, 0]
    # 取exp_name为default的before_reward列
    srr_default = combined_df[combined_df["exp_name"] == "default"]["SRR"].tolist()
    srr_trick = combined_df[combined_df["exp_name"] == trick_name]["SRR"].tolist()

    # Plotting the line chart with std as the shaded area
    golden_ratio = 1.618
    width = 14  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))
    fig, ax = plt.subplots(figsize=(width, height))
    metrics_hat_graph(
        ax,
        xlabels,
        [start, srr_default, srr_trick],
        ["default", f"SRR with default setting", f"SRR with {trick_name}"],
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Environments")
    ax.set_ylabel("Change Rate")
    ax.set_title("Self-robustness realated metrics in different environments of MAPPO")
    ax.set_ylim(-1, 0.2)
    # 设置Y轴为百分比格式
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()

    fig.tight_layout()

    # 保存图表到文件
    category = "necessity"
    save_dir = os.path.join(argv["out"], "figures", argv["i18n"], argv["type"], category)
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(save_dir, f'{trick_name}_self_metrics.{argv["type"]}')
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figure_name}")


def draw_trick_metrics_hat(combined_df, trick_name, argv):
    # xlabels为combined_df的scenario列，去重
    xlabels = combined_df["scenario"].unique().tolist()
    start = [0, 0, 0, 0]
    # 取exp_name为default的before_reward列
    rsrr = combined_df[combined_df["exp_name"] == trick_name]["rSRR"].tolist()
    tpr = combined_df[combined_df["exp_name"] == trick_name]["TPR"].tolist()
    trr = combined_df[combined_df["exp_name"] == trick_name]["TRR"].tolist()

    # Plotting the line chart with std as the shaded area
    golden_ratio = 1.618
    width = 14  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))
    fig, ax = plt.subplots(figsize=(width, height))
    metrics_hat_graph(
        ax,
        xlabels,
        [start, rsrr, tpr, trr],
        ["default", f"rSRR with {trick_name}", f"TPR with {trick_name}", f"TRR with {trick_name}"],
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Environments")
    ax.set_ylabel("Change Rate")
    ax.set_title("Trick-robustness realated metrics in different environments of MAPPO")
    # ax.set_ylim(-0.25, 0.25)
    # 设置Y轴为百分比格式
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()

    fig.tight_layout()

    # 保存图表到文件
    category = "necessity"
    save_dir = os.path.join(argv["out"], "figures", argv["i18n"], argv["type"], category)
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(save_dir, f'{trick_name}_trick_metrics.{argv["type"]}')
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figure_name}")


def hat_graph(ax, xlabels, values, group_labels):
    """
    Create a hat graph.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes to plot into.
    xlabels : list of str
        The category names to be displayed on the x-axis.
    values : (M, N) array-like
        The data values.
        Rows are the groups (len(group_labels) == M).
        Columns are the categories (len(xlabels) == N).
    group_labels : list of str
        The group labels displayed in the legend.
    """

    def label_bars(heights, rects):
        """Attach a text label on top of each bar."""
        for height, rect in zip(heights, rects):
            # 保留两位小数
            height = round(height, 2)
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),  # 4 points vertical offset.
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    values = np.asarray(values)
    x = np.arange(values.shape[1])
    ax.set_xticks(x, labels=xlabels)
    spacing = 0.3  # spacing between hat groups
    width = (1 - spacing) / values.shape[0]
    heights0 = values[0]
    for i, (heights, group_label) in enumerate(zip(values, group_labels)):
        style = {"fill": False} if i == 0 else {"edgecolor": "black"}
        rects = ax.bar(
            x - spacing / 2 + i * width, heights - heights0, width, bottom=heights0, label=group_label, **style
        )
        label_bars(heights, rects)

def metrics_hat_graph(ax, xlabels, values, group_labels):
    """
    Create a hat graph.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes to plot into.
    xlabels : list of str
        The category names to be displayed on the x-axis.
    values : (M, N) array-like
        The data values.
        Rows are the groups (len(group_labels) == M).
        Columns are the categories (len(xlabels) == N).
    group_labels : list of str
        The group labels displayed in the legend.
    """

    def label_bars(heights, rects):
        """Attach a text label on top of each bar."""
        for height, rect in zip(heights, rects):
            # 保留两位小数
            height = round(height, 4)
            ax.annotate(
                f"{height:.2%}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),  # 4 points vertical offset.
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    values = np.asarray(values)
    x = np.arange(values.shape[1])
    ax.set_xticks(x, labels=xlabels)
    spacing = 0.3  # spacing between hat groups
    width = (1 - spacing) / values.shape[0]
    heights0 = values[0]
    for i, (heights, group_label) in enumerate(zip(values, group_labels)):
        style = {"fill": False} if i == 0 else {"edgecolor": "black"}
        rects = ax.bar(
            x - spacing / 2 + i * width, heights - heights0, width, bottom=heights0, label=group_label, **style
        )
        label_bars(heights, rects)


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
                f"{argv['env']}_{argv['scenario']}_{argv['algo']}.xlsx",
            )

        # # 评一个trick方案下所有攻击的reward
        # # x轴为trick，y轴为reward，每个trick方案一张图
        # plot_trick_reward(excel_path, argv)
        # if argv["env"] == "smac":
        #     plot_trick_reward(excel_path, argv, ylabel="Win Rate")

        # # 评一个攻击下所有trick的reward
        # # x轴为trick，y轴为reward，每个攻击方法一张图
        # plot_attack_reward(excel_path, argv)

        # # 画metrics
        # plot_metrics(excel_path, argv)

        # # 合并attack、metrics的箱线图，每个（环境+算法）一张图，共12张，看的是算法在特定环境上的不同trick的表现
        # boxplot_mean_attack_metric(excel_path, argv)

        # # 合并attack，画metrics的errorbar图
        # errorbar_mean_attack_metric(excel_path, argv)
        # # 合并attack，画metrics的bar图，带着std
        # bar_mean_attack_metric(excel_path, argv)

    # # 画训练对比曲线图
    # plot_train_reward(argv)

    # # 合并env、attack、metrics的箱线图，每个算法一张图，共3张，看的是算法在所有环境上的不同trick的表现
    # boxplot_mean_attack_metric_env(argv)

    # 画metric必要性分析图
    plot_necessity(argv)
