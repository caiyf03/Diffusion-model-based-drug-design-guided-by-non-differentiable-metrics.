import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# 1. 你的文件路径列表
files = [
    "DiffSBDD/my_example_inpaint/origion_50_new/out.txt",
    "DiffSBDD/my_example_inpaint/SPSA_50_new/out.txt",
    "DiffSBDD/my_example_inpaint/SVDD_50_new/out.txt",
]

# 2. 解析每行指标
records = []
pattern = re.compile(
    r"QED: ([0-9.]+) \+/- [0-9.]+, SA: ([0-9.]+) \+/- [0-9.]+, "
    r"LogP: ([-0-9.]+) \+/- [0-9.]+, Lipinski: ([0-9.]+)"
)
for fname in files:
    model = os.path.basename(os.path.dirname(fname))
    with open(fname, "r") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            qed, sa, logp, lip = map(float, m.groups())
            records.append({
                "Model": model,
                "QED":   qed,
                "SA":    sa,
                "LogP":  logp,
                "Lipinski": lip,
            })

df = pd.DataFrame(records)

# 3. 自定义任务名列表，顺序要和下面 models 一致
task_names = [
    "Origion (DiffSBDD)",    # 对应 "600_final"
    "SPSA",
    "ATP", 
]

# 4. 每个指标的 y 坐标范围
y_limits = {
    "QED":      (0, 1.1),
    "SA":       (0, 1.1),
    "LogP":     (-5, 10),
    "Lipinski": (0, 5.1),
}

sns.set_style("whitegrid")
dpi = 300

# 5. 获取所有 model 的顺序
models = df["Model"].unique().tolist()
palette_colors = [
    "#D3D3D3",  # 浅灰
    "#FFFACD",  # 浅黄 (LemonChiffon)
    "#ADD8E6",  # 浅蓝 (LightBlue)
]
name = ["QED from 0 to 1, and the higher the better", "SA from 0 to 1, and the higher the better", "LogP from -5 to 10, with [-1,5] assumed to be good", "Lipinski from 0 to 5, and the higher the better"]
count = 0
for metric, (ymin, ymax) in y_limits.items():
    plt.figure(figsize=(6, 4), facecolor="white")
    ax = sns.violinplot(
        x="Model", y=metric, data=df,
        scale="count",    # 宽度按样本数缩放
        inner=None,       # 取消内部统计线
        palette=palette_colors
    )

    # 用自定义的 task_names 替换横坐标
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(task_names, rotation=0)

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(f"{name[count]}")
    ax.set_ylabel(metric)
    ax.set_title(f" violin plot: width ∝ probability density")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    plt.tight_layout()
    out_path = f"DiffSBDD/my_pic_inpaint/{metric}_violin_tasks.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    count += 1
