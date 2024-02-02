import os

from matplotlib import pyplot as plt

from utils.plot.config import i18n
from utils.plot.colors import line_colors, line_markers


def line_tricks(df, title, ylabel, figurename, argv):
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
    plt.title(f"{argv['env']}_{argv['scenario']}_{argv['algo']}_{title}")
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
    save_dir = os.path.dirname(figurename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(figurename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figurename}")

    # 展示图表
    if argv["show"]:
        plt.show()