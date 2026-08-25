# AI Toolbox For Research

> **15 章 · 350 页 · 83 个 Python 代码示例**
>
> 从 "什么是 AI？" 到 "这是我的完整研究论文" —— 一本面向学生和初学者的 AI 研究实战教材。

---

## 📖 项目简介

《AI Toolbox For Research》是一本**项目驱动**的 AI 研究入门教材，不要求读者有任何编程或高等数学基础。全书围绕一个核心案例展开：**自动驾驶感知（Teaching a Car to See）**——如何让汽车从摄像头画面中识别行人、车辆和交通标志。

本仓库包含：
- 📕 **教材源码**（LaTeX，可编译为 PDF）
- 🐍 **全部代码示例**（已提取到单一 Python 文件）
- 📊 **15 章配套 PPT**（每章一份）
- 📚 **参考文献库**（BibTeX）

---

## 📁 仓库结构

```
.
├── README.md                  # 本文件
├── main.tex                   # 教材完整源码（单文件，约 677 KB）
├── bibliography.bib           # 参考文献数据库（BibTeX，53 条）
├── all_code_listings.py       # 全书 83 个 Python 代码示例（已提取合并）
├── PPT/                       # 15 章配套幻灯片（.pptx）
│   ├── AI Toolbox For Research - Chapter 1.pptx
│   ├── AI Toolbox For Research - Chapter 2.pptx
│   ├── ...
│   └── AI Toolbox For Research - Chapter 15.pptx
├── ChatGPT Image1.png         # 封面图
├── ChatGPT Image 2.png        # 封底图
├── fig_ch9_*.png              # 第 9 章图表（直方图/条形图/散点图/热力图）
├── av_confidence_scatter.png  # 第 9 章案例 1 图表
└── ch9_chart3_heatmap.png     # 第 9 章案例 2 图表
```

---

## 🐍 代码结构（all_code_listings.py）

全书 **83 个 Python 代码示例**已合并到单一文件 `all_code_listings.py`，便于查阅和复制。

### 组织方式

每个代码块用注释分隔，标注原始 caption：

```python
# ============================================================================
# Listing 1: Exploring the AV perception problem --- the first step in any AI project
# ============================================================================

import numpy as np
import pandas as pd
...
```

### 代码清单分布

| 章节 | 主题 | 主要代码示例 |
|------|------|-------------|
| 第 1 章 | 什么是 AI | AV 对象分类、规则 vs 机器学习对比 |
| 第 2 章 | 什么是研究 | 研究问题三测试评分、AV 研究问题框架 |
| 第 3 章 | AI 研究管线 | Iris 端到端管道、AV 数据集对比 |
| 第 4 章 | LLM 做研究 | 提示工程、LLM 论文摘要验证 |
| 第 5 章 | Python 数据分析 | Colab、pandas、NumPy、matplotlib、seaborn |
| 第 6 章 | 研究工具箱 | 数据集平台、模型工具、实验平台 |
| 第 7 章 | 什么是数据 | AV 数据集导览、Kaggle 数据集速查 |
| 第 8 章 | 数据清洗 | Airbnb 清洗、AV 检测日志清洗 |
| 第 9 章 | 数据可视化 | 四类核心图表、AV 数据故事 |
| 第 10 章 | 机器学习 | 线性回归、决策树、随机森林、XGBoost |
| 第 11 章 | 深度学习 | CNN、RNN/LSTM、Transformer、MLP |
| 第 12 章 | 大语言模型 | Transformer、RAG、LoRA、Agent |
| 第 13 章 | 多智能体 | LangGraph、AutoGen、CrewAI、MetaGPT |
| 第 14 章 | 实验设计 | Baseline/对比/消融/参数实验 |
| 第 15 章 | 研究展示 | 论文写作、模型迷你研究 |
| 附录 | 环境配置、快速参考、习题解答、术语表 | 安装、EDA 模板、KNN 调参等 |

> ⚠️ **注意**：部分代码块假设同一节的**前序代码已运行**（如变量 `df`、`X_train` 等），请按顺序运行。

---

## 📊 配套 PPT

`PPT/` 目录包含 15 章配套幻灯片（`.pptx` 格式），每章一份，内容与教材对应。

| 文件 | 对应章节 |
|------|---------|
| `Chapter 1.pptx` | 什么是 AI？ |
| `Chapter 2.pptx` | 什么是研究？ |
| `Chapter 3.pptx` | AI 研究管线 |
| `Chapter 4.pptx` | LLM 做研究 |
| `Chapter 5.pptx` | Python 数据分析 |
| `Chapter 6.pptx` | 研究工具箱 |
| `Chapter 7.pptx` | 什么是数据？ |
| `Chapter 8.pptx` | 数据清洗 |
| `Chapter 9.pptx` | 数据可视化 |
| `Chapter 10.pptx` | 机器学习 |
| `Chapter 11.pptx` | 深度学习 |
| `Chapter 12.pptx` | 大语言模型 |
| `Chapter 13.pptx` | 多智能体 |
| `Chapter 14.pptx` | 实验设计 |
| `Chapter 15.pptx` | 研究展示 |

---

## 🔧 编译教材

### 本地编译（推荐）

```bash
pdflatex -interaction=nonstopmode main.tex
biber main
makeindex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### Overleaf 编译

1. 上传 `main.tex`、`bibliography.bib` 及所有图片文件
2. 将 Compiler 设为 **pdfLaTeX**
3. Overleaf 会自动执行 biber/makeindex

---

## 👥 作者

- **Xiwang Guo** — Visiting Associate Professor, NJIT；Associate Professor, Liaoning Petrochemical University
- **Jiangtao Cao** — Full Professor, School of Information and Control Engineering, Liaoning Petrochemical University
- **Chengbo Hu** — Director, Publicity Department, CPC Liaoning Provincial Committee Education Working Committee

---

## 📄 许可

Creative Commons Zero 1.0 Universal (CC0) —— 自由使用、修改和分发。
