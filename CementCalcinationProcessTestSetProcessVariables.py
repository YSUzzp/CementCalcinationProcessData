import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# ---------------------- 关键设置：基础配置 ----------------------
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['figure.facecolor'] = 'white'  # 图表背景白色
plt.rcParams['legend.framealpha'] = 0.95  # 图例背景透明度
plt.rcParams['legend.handlelength'] = 1.2  # 图例线条长度
plt.rcParams['legend.labelspacing'] = 0.4  # 图例标签间距


# ---------------------- 文件夹创建与路径管理 ----------------------
def create_save_folder(base_path):
    """创建保存文件夹，存在则不报错，返回文件夹路径"""
    folder_name = f"一次异常发生的时间序列测试集(44200-44560)"
    save_folder = os.path.join(base_path, folder_name)
    os.makedirs(save_folder, exist_ok=True)
    print(f"📂 保存文件夹：{save_folder}")
    return save_folder


# 基础保存路径（当前程序所在目录）
BASE_SAVE_PATH = "."  # 改为当前目录，也可根据需要调整
SAVE_FOLDER = create_save_folder(BASE_SAVE_PATH)

# ---------------------- 核心配置：变量名+单位映射（精准匹配需求） ----------------------
# 变量配置：中文名称 → (英文名称, 单位)
VAR_CONFIG = {
    '尾排风机变频电流反馈': ('Tail Fan Inverter Current Feedback', 'A'),
    '铝氧率平均值': ('Average Aluminum-Oxygen Ratio', '%'),
    '窑头排风机工频电流': ('Kiln Head Fan Power Frequency Current', 'A'),
    '吨熟料标煤耗': ('Standard Coal Consumption per Ton of Clinker', 'kgce/t'),
    'C4B出口温度': ('C4B Outlet Temperature', '℃'),
    '窑电流平均值': ('Average Kiln Current', 'A'),
    '喂料量反馈': ('Feeding Rate Feedback', 't/h'),
    '窑尾温度平均值': ('Average Kiln Outlet Temperature', '℃'),
    'C2A出口压力': ('C2A Outlet Pressure', 'Pa'),
    '高温风机入口气体温度': ('High-Temp Fan Inlet Gas Temperature', '℃'),
    '高温风机入口气体压力': ('High-Temp Fan Inlet Gas Pressure', 'Pa'),
    '分解炉煤反馈': ('Calciner Coal Feed Feedback', 'kg/h'),
    '二次风温平均值': ('Average Secondary Air Temperature', '℃'),
    '窑尾负压': ('Kiln Outlet Negative Pressure', 'Pa'),
    '三次风温': ('Tertiary Air Temperature', '℃'),
    '窑喂料提升机电流': ('Kiln Feeding Hoist Current', 'A'),
    '窑系统单位电耗': ('Kiln System Unit Power Consumption', 'kWh/t'),
    'C4A出口温度': ('C4A Outlet Temperature', '℃'),
    'C2A出口温度': ('C2A Outlet Temperature', '℃'),
    'f_CaO': ('f_CaO Content', '%')
}

# 目标列（中文）
target_cols_cn = list(VAR_CONFIG.keys())
# 英文列名+单位+序号（用于图例）
target_cols_en = [VAR_CONFIG[col][0] for col in target_cols_cn]
var_units = [VAR_CONFIG[col][1] for col in target_cols_cn]
# 英文图例标签：简化显示（避免过长）
columns_en_legend = [f"X{i + 1}: {en_name.split(' ')[0] if len(en_name.split(' ')) > 1 else en_name} [{unit}]"
                     for i, (en_name, unit) in enumerate(zip(target_cols_en, var_units))]

# ---------------------- 1. 读取数据并筛选 ----------------------
# 修改为同级别目录下的相对路径
CSV_PATH = "CementCalcinationProcessData.csv"

# 读取数据
data50 = pd.read_csv(CSV_PATH, encoding="gbk")
print(f"\n📥 成功读取原始数据: 形状{data50.shape} (行×列)")

# 验证目标列是否存在
missing_cols = [col for col in target_cols_cn if col not in data50.columns]
if missing_cols:
    print(f"⚠️  警告：以下列在CSV中不存在，请检查列名：{missing_cols}")
else:
    print("✅ 所有目标列均存在")

# 筛选数据（索引<=80000）
filtered_data_all = data50.loc[data50.index <= 80000, target_cols_cn]
print(f"\n🔍 筛选后数据形状: {filtered_data_all.shape} (行×列)")

# 截取原始行号44200-44560（0开始索引：44199-44559，共361行）
start_idx = 44200 - 1
end_idx = 44560 - 1
df = filtered_data_all.iloc[start_idx:end_idx + 1, :].reset_index(drop=True)

print(f"🎯 截取目标区间（原始行号44200-44560）: 形状{df.shape} (行×列)")
if df.empty:
    print("⚠️  警告：截取的数据为空！请检查筛选后的数据索引范围")
    print(f"   筛选后数据索引范围：0 ~ {filtered_data_all.index.max()}")
else:
    print("✅ 数据截取成功")

# ---------------------- 2. 打印变量对照表 ----------------------
print("\n📋 变量中英文+单位对照表:")
print(f"{'序号':<6}{'中文名':<18}{'英文名':<40}{'单位':<8}")
print("-" * 70)
for i, (cn, en, unit) in enumerate(zip(target_cols_cn, target_cols_en, var_units), 1):
    print(f"X{i:<5}{cn:<18}{en:<40}{unit:<8}")


# ---------------------- 3. 通用绘图函数（优化图例不遮挡） ----------------------
def plot_time_series(df, legend_labels, title, xlabel, ylabel, save_filename, is_english=False):
    """
    绘制时间序列图（2列10行布局），优化图例不遮挡曲线
    :param df: 数据框
    :param legend_labels: 图例标签列表
    :param title: 图表总标题
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param save_filename: 保存文件名
    :param is_english: 是否为英文图（控制字体、图例、对齐）
    """
    # 调整图表大小：英文图预留更多右侧空间
    if is_english:
        fig_width = 22  # 英文图更宽，容纳长图例
        subplot_left = 0.06
        subplot_right = 0.82  # 预留右侧18%空间放图例
    else:
        fig_width = 19
        subplot_left = 0.08
        subplot_right = 0.88  # 预留右侧12%空间放图例

    fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(fig_width, 32), squeeze=False)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.99, alpha=0.95)  # 总标题高透明度

    # 颜色列表（20种鲜明颜色）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
              '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7',
              '#dbdb8d', '#9edae5']

    # 绘制每个变量
    for idx, (col, legend_label, color) in enumerate(zip(df.columns, legend_labels, colors)):
        row = idx // 2  # 0-9（10行）
        col_idx = idx % 2  # 0-1（2列）
        ax = axes[row, col_idx]

        # 绘制时间序列（英文图线条略粗）
        line_width = 2.0 if is_english else 1.8
        ax.plot(df.index, df[col], color=color, linewidth=line_width, alpha=0.9, label=legend_label)

        # ---------------------- 英文图核心优化 ----------------------
        if is_english:
            # 全局字体：Times New Roman
            font_config = {'fontfamily': 'Times New Roman', 'fontsize': 12, 'alpha': 0.95}
            legend_font_size = 11  # 英文图例字体稍小，避免过长

            # 图例放在子图右侧外部，不遮挡曲线
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),  # 锚点在子图右侧中间
                      frameon=True, fancybox=True, shadow=False,
                      prop={'family': 'Times New Roman', 'size': legend_font_size},
                      framealpha=0.95,
                      labelspacing=0.3, handlelength=1.2, handletextpad=0.5)

        # ---------------------- 中文图优化（保留但不再调用） ----------------------
        else:
            # 中文配置：字体+字号
            font_config = {
                'fontfamily': ['SimHei', 'Microsoft YaHei'],
                'fontsize': 11,
                'fontweight': 'bold',
                'alpha': 0.95
            }
            legend_font_size = 10

            # 图例放在子图右侧外部，不遮挡曲线
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),  # 锚点在子图右侧中间
                      frameon=True, fancybox=True, shadow=False,
                      prop={'family': ['SimHei', 'Microsoft YaHei'], 'size': legend_font_size},
                      framealpha=0.95,
                      labelspacing=0.2, handlelength=1.0, handletextpad=0.3)

            # 子图标题（中文图保留子图标题）
            ax.set_title(legend_label.split(' (')[0], pad=10, **font_config)

        # 坐标轴标签
        ax.set_xlabel(xlabel, **font_config)
        ax.set_ylabel(ylabel, **font_config)

        # 刻度标签
        if is_english:
            ax.tick_params(axis='both', which='major', labelsize=11,
                           labelfontfamily='Times New Roman')
        else:
            ax.tick_params(axis='both', which='major', labelsize=10,
                           labelfontfamily=['SimHei', 'Microsoft YaHei'])

        # 仅最下面一行显示完整x轴标签
        if row != 9:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)

        # 网格设置
        grid_linewidth = 1.0 if is_english else 0.8
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=grid_linewidth)

        # 优化y轴范围（避免曲线贴边）
        y_min = df[col].min()
        y_max = df[col].max()
        y_margin = (y_max - y_min) * 0.08 if y_max != y_min else 0.15
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # 调整坐标轴标签位置，避免重叠
        ax.xaxis.labelpad = 8
        ax.yaxis.labelpad = 8

    # 调整子图间距：为右侧图例留出空间
    plt.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        top=0.97,
        bottom=0.03,
        hspace=0.3,  # 行间距
        wspace=0.5  # 列间距（英文图更大）
    )

    # 保存图片（防覆盖）
    save_path = os.path.join(SAVE_FOLDER, save_filename)
    if os.path.exists(save_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        name, ext = os.path.splitext(save_filename)
        save_path = os.path.join(SAVE_FOLDER, f"{name}_{timestamp}{ext}")
        print(f"⚠️  文件已存在，添加时间戳：{os.path.basename(save_path)}")

    # 高分辨率保存，确保图例清晰
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    print(f"✅ 图表保存成功：{save_path}")

    # 显示图表
    plt.show()


# ---------------------- 4. 生成英文版本图表（仅保留此项） ----------------------
if not df.empty:
    print("\n🎨 生成英文版本图表...")
    # 创建英文列名的数据框
    df_en = df.copy()
    df_en.columns = target_cols_en

    plot_time_series(
        df=df_en,
        legend_labels=columns_en_legend,  # 简化后的英文图例，避免过长
        title='Cement calcination process test set:process variables',  # 空标题，可根据需要修改
        xlabel='Index in Interval (0-360)',
        ylabel='Value',
        save_filename='Cement_Process_Test_English_44200-44560.png',
        is_english=True
    )

# ---------------------- 5. 输出数据统计信息 ----------------------
if not df.empty:
    print("\n📊 目标区间数据基本统计信息（中文）:")
    stats_cn = df.describe().T
    stats_cn.index = [f"{col} (X{i + 1})" for i, col in enumerate(target_cols_cn)]
    print(stats_cn.round(3))

    print("\n📊 Basic Statistical Information (English):")
    stats_en = df_en.describe().T
    stats_en.index = [f"X{i + 1}: {name} [{unit}]" for i, (name, unit) in enumerate(zip(target_cols_en, var_units))]
    print(stats_en.round(3))
else:
    print("\n❌ 无法生成可视化：目标区间数据为空")