import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macos font
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # linux
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题
# 定义马卡龙配色方案的颜色
macaron_colors_1 = ['#83C5BE', '#FFDDD2', '#FFBCBC', '#FFAAA5', '#F8BBD0', '#FF8C94']
# 柔和粉红色 (Soft Pink), 天空蓝色 (Sky Blue),淡紫罗兰色 (Lavender),薄荷绿色 (Mint Green),淡黄色 (Light Yellow),杏色 (Apricot),淡橙色 (Light Orange),淡绿色 (Pale Green),淡蓝色 (Pale Blue),淡紫色 (Pale Purple)
macaron_colors_2 = ['#F8BBD0', '#81D4FA', '#B39DDB', '#C8E6C9', '#FFF59D', '#FFCC80', '#FFAB91', '#C5E1A5', '#80DEEA', '#CE93D8']
# https://blog.csdn.net/slandarer/article/details/114157177
macaron_colors_3 = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2', '#999999']
ray_colors = ["#83C5BE", "#FFDDD2", "#FFAB91", "#FFAB40", "#FFE0B2", "#FF8C94", '#999999']

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
        "E": "多智能体特性"
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
        "E": "Multi-Agent Feature"
    },
}

YLIM = {
    "smac_2s3z_mappo": [0, 35],
    "smac_2s3z_qmix": [0, 35],
    "smac_3m_mappo": [0, 25],
    "smac_3m_qmix": [0, 25],
    "mamujoco_HalfCheetah-6x1_mappo": [-10000, 25000],
}

# 读取scheme.json
SCHEME_CFG = json.load(open("settings/scheme.json", "r"))

def plot_trick_reward(excel_path, argv):
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
            combined_df["vanilla_reward"] = df["vanilla_reward"]
        
        # 对combined_df增加一列，以sheet作为列名，值为df中的adv_reward
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
        _plot_line(plot_data, excel_path, "trick", name, argv)


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
        plt.bar(bar_positions[i], df[column], color=ray_colors[i], width=bar_width, edgecolor='grey', label=display[column])

    # 添加图表细节
    # 判断YLIM是否有该filename的key，如果有，则设置Y轴范围
    if filename in YLIM:
        plt.ylim(YLIM[filename])
    # plt.xlabel(display[scheme[0]])
    plt.xticks(positions + total_bar_space / 2, df['exp_name'], rotation=0 if len(df) < 10 else 45, ha="right")
    plt.ylabel(display["reward"])
    # 在y=0处添加一条水平线
    plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    # plt.title(scheme_name)
    plt.legend(ncol=10, frameon=True, loc='upper center', bbox_to_anchor=(0.5, 1))
    plt.tight_layout()

    # 保存图表到文件
    # ./out/figures/en/png/smac/3m/mappo/smac_3m_mappo_A1.png
    save_dir = os.path.join(argv["out"], "figures", argv["i18n"], argv["type"], argv["env"], argv["scenario"], argv["algo"], category)
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(save_dir, f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}_{category}_{name}.{argv["type"]}')
    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
    print(f"Saved to {figure_name}")

    # 展示图表
    if argv["show"]:
        plt.show()


def _plot_line(df, excel_path, category, name, argv):
    filename = os.path.basename(excel_path).split(".")[0]
    display = i18n[argv["i18n"]]

    # 新画布
    golden_ratio = 1.618
    width = 15  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))

    # 定义不同的点形状和颜色
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'x', '*', '+', '1', '2', '3', '4']
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'pink', 'brown', 'gray', 'purple', 'olive', 'cyan', 'navy', 'teal']

    # 绘制每一行的数据
    # 将df列名按照i18n进行替换
    df.columns = [display[col] if col in display else col for col in df.columns]
    for i, row in df.iterrows():
        plt.plot(row.index[1:], row.values[1:], linestyle='--', marker=markers[i], color=colors[i], label=row[display["exp_name"]])

    # 设置图表标题和坐标轴标签
    # plt.title('Rewards for Different Attack Methods')
    # plt.xlabel('Attack Methods')
    plt.ylabel('Reward')

    # 判断YLIM是否有该filename的key，如果有，则设置Y轴范围
    # if filename in YLIM:
    #     plt.ylim(YLIM[filename])

    # 添加网格虚线
    plt.grid(True, linestyle='--')

    # 显示图例
    plt.legend()

    # 保存图表到文件
    # ./out/figures/en/png/smac/3m/mappo/smac_3m_mappo_A1.png
    save_dir = os.path.join(argv["out"], "figures", argv["i18n"], argv["type"], argv["env"], argv["scenario"], argv["algo"], category)
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(save_dir, f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}_{category}_{name}.{argv["type"]}')
    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
    print(f"Saved to {figure_name}")

    # 展示图表
    if argv["show"]:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="smac", help="env name")
    parser.add_argument("-s", "--scenario", type=str, default="3m", help="scenario or map name")
    parser.add_argument("-a", "--algo", type=str, default="mappo", help="algo name")
    parser.add_argument("-o", "--out", type=str, default="out", help="output dir")
    parser.add_argument('-f', '--file', type=str, default=None, help='Excel file path')
    parser.add_argument('-i', '--i18n', type=str, default='en', choices=["en", "zh"], help='Choose the language')
    parser.add_argument("-t", "--type", type=str, default='png', choices=["png", "pdf"], help='save figure type')
    parser.add_argument('-g', '--groupby', type=str, default='trick', choices=["trick", "scheme"], help='Group by trick or scheme')
    parser.add_argument('--show', action='store_true', help='Whether to show the plot')
    args, _ = parser.parse_known_args()
    argv = vars(args)
    
    if argv["file"] is not None:
        excel_path = argv["file"]
    else:
        excel_path = os.path.join(argv["out"], "data", argv["env"], argv["scenario"], argv["algo"], f"{argv['env']}_{argv['scenario']}_{argv['algo']}.xlsx")

    # 评一个trick方案下所有攻击的reward
    # x轴为trick，y轴为reward，每个trick方案一张图
    plot_trick_reward(excel_path, argv)

    # 评一个攻击下所有trick的reward
    # x轴为trick，y轴为reward，每个攻击方法一张图
    plot_attack_reward(excel_path, argv)