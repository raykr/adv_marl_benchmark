import os
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from utils.plot.config import BOXPLOT_YLIM
from utils.plot.colors import rainbow_colors_2


def boxplot_cr(row_wise_results, title, figurename):
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

    algo_name = title.split("_")[-1]
    if title in BOXPLOT_YLIM:
        plt.ylim(BOXPLOT_YLIM[algo_name])
    # 在y=0处添加一条水平线
    plt.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    # 设置Y轴为百分比格式
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.grid(True, axis='both', linestyle='--')
    plt.xticks(positions, exp_names, rotation=35, ha="right")
    plt.xlabel("Tricks")
    plt.ylabel("Comprehensive Robustness Change Rate")
    plt.title(title)
    plt.tight_layout()

    # 保存图表到文件
    save_dir = os.path.dirname(figurename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(figurename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figurename}")