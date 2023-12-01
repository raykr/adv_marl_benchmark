import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import PercentFormatter
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macos font
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题

parser = argparse.ArgumentParser(description="Analysis AMB Experiments Data.")
parser.add_argument("--file", type=str, default="HalfCheetah-MAPPO-ADV.xlsx", help="Please enter the path of .xlsx.")
parser.add_argument("--save", action="store_true", help="If save the plot.")
parser.add_argument("--show", action="store_true", help="If show the plot.")
argv = vars(parser.parse_args())
print(argv)


# 获取Excel文件中所有sheet的名称
xls = pd.ExcelFile(argv["file"])
sheet_names = xls.sheet_names
# 获取文件名
file_name = os.path.basename(argv["file"]).split(".")[0]

# 定义马卡龙配色方案的颜色
macaron_colors = ['#83C5BE', '#FFDDD2', '#FFBCBC', '#FFAAA5', '#FF8C94']

# 遍历所有sheets
for sheet_name in sheet_names:
    # 读取当前sheet
    data_sheet = pd.read_excel(argv["file"], header=[0, 1], sheet_name=sheet_name)

    # 从数据中选择需要的列
    selected_columns = data_sheet[[
        ('实现细节', 'Unnamed: 5_level_1'),
        ('以Default模型为Baseline', '(R - baseline)/baseline\ntrick性能提升率'),
        ('以Default模型为Baseline', '(Ra - baseline_a)/baseline\nTrick鲁棒性提升率'),
        ('以自身模型为Baseline', '(Ra - R)/R\n自身鲁棒性')
    ]].copy()
    selected_columns.columns = ['实现细节', 'Trick性能提升率', 'Trick鲁棒性提升率', '自身鲁棒性']

    # 绘制柱状图
    x_labels = selected_columns['实现细节']
    self_robustness = selected_columns['自身鲁棒性']
    trick_performance_increase = selected_columns['Trick性能提升率']
    trick_robustness_increase = selected_columns['Trick鲁棒性提升率']
    n_categories = len(selected_columns)

    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.25
    r1 = np.arange(n_categories)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    ax.bar(r1, self_robustness, color=macaron_colors[0], width=bar_width, edgecolor='grey', label='Self Robustness')
    ax.bar(r2, trick_performance_increase, color=macaron_colors[1], width=bar_width, edgecolor='grey', label='Performance Increase Rate')
    ax.bar(r3, trick_robustness_increase, color=macaron_colors[4], width=bar_width, edgecolor='grey', label='Adv Robustness Increase Rate')

    # 在y=0处添加一条水平线
    ax.axhline(y=0, color='black', linewidth=1.5)

    # 设置Y轴为百分比格式
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    # ax.set_xlabel('实现细节')
    ax.set_ylabel('Reward change rate')
    ax.set_title(f'{sheet_name}')
    ax.set_xticks([r + bar_width for r in range(n_categories)])
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    # 显示图表
    if argv["show"]:
        plt.show()

    if argv["save"]:
        # 判断输出路径是否存在
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        
        # 替换掉sheet_name中的（）和中文
        sheet_name = sheet_name.replace('（', '').replace('）', '').replace(' ', '-').replace('：', '').replace('，', '')
        # 去除sheet_name中的中文字符
        sheet_name = ''.join([i for i in sheet_name if not ('\u4e00' <= i <= '\u9fa5')])

        # 保存图表
        fig.savefig(f'{file_name}/{file_name}-{sheet_name}.png', dpi=300)
        print(f'Save {file_name}/{file_name}-{sheet_name}.png successfully.')


