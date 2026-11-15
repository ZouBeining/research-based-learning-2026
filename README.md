# 文本分类任务中线性分类器的基线作用研究

高中研究性学习论文项目。参考论文：

> Yu-Chen Lin, Si-An Chen, Jie-Jyun Liu, Chih-Jen Lin.
> *Linear Classifier: An Often-Forgotten Baseline for Text Classification.*
> ACL 2023 (Short Papers), pp. 1876–1888.
> [arXiv:2306.07111](https://arxiv.org/abs/2306.07111)

参考论文的核心观点是：在文本分类任务中，简单的线性分类器（TF-IDF + 逻辑回归 / 线性 SVM）
常被忽视，但它们往往能以极低代价取得与预训练语言模型相近的效果，
并且是检验复杂模型配置是否正确的必要基线。

本项目在中英文 4 个公开数据集上复现并检验这一观点。

---

## 目录结构

```
研究性学习/
├── AGENTS.md                    论文写作要求
├── README.md                    本文件
├── paper/                       LaTeX 论文
│   ├── main.tex                 主文件（xelatex 编译）
│   ├── references.bib           BibTeX 参考文献
│   └── figures/                 自动生成的 pgfplots 图表
│       ├── f1_comparison.tex
│       └── train_time.tex
└── experiment/                  实验代码
    ├── dataset_downloader.py    数据集下载与缓存
    ├── data_utils.py            数据加载、切分、特征提取
    ├── models.py                线性模型与神经网络模型定义
    ├── run_experiments.py       主实验脚本
    ├── make_paper_assets.py     由结果 JSON 生成论文表格与图表
    ├── data/                    数据集缓存（JSONL，自动下载）
    └── results/                 实验结果（CSV / JSON / LaTeX 表格）
```

---

## 环境准备

需要 Python 3.10 以上版本。依赖仅为三个常用库：

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install torch scikit-learn numpy
```

实验在 **CPU 上即可完成**，全程约 15 分钟，无需 GPU。

---

## 复现步骤

```bash
cd experiment

# 1) 下载数据集（首次运行，约 8 分钟；之后走本地缓存）
python dataset_downloader.py --all

# 2) 运行全部实验（约 15 分钟）
python run_experiments.py

# 3) 由结果 JSON 生成论文表格与图表
python make_paper_assets.py
```

编译论文：

```bash
cd ../paper
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
```

### 常用选项

```bash
# 快速模式：小数据、少轮次，约 2 分钟（用于验证流程是否跑通）
python run_experiments.py --quick

# 只跑部分数据集 / 部分模型
python run_experiments.py --datasets ag_news tnews
python run_experiments.py --models "TF-IDF + Linear SVM" TextCNN
```

---

## 数据集

全部为公开数据集，脚本会自动下载并缓存到 `experiment/data/`。

| 数据集 | 语言 | 类别数 | 规模 | 来源 |
|---|---|---|---|---|
| AG News | 英文 | 4 | 7 819 | `fancyzhx/ag_news` |
| 20 Newsgroups | 英文 | 4 | 2 324 | `sklearn.datasets.fetch_20newsgroups` |
| TNews | 中文 | 10 | 4 999 | `C-MTEB/TNews-classification`（CLUE 基准） |
| ChnSentiCorp | 中文 | 2 | 7 977 | `lansinuote/ChnSentiCorp` |

> 数据集通过 HuggingFace `datasets-server` 公开 REST API 下载，无需登录或 API Key。
> 若网络无法访问该服务，可自行下载原始数据后按
> `data/*.jsonl` 的格式（每行 `{"text": ..., "label": ...}`，首行为元信息）放置。

---

## 对比的模型

**线性模型（scikit-learn）**

- `TF-IDF + Logistic Regression`：多项逻辑回归，softmax 输出
- `TF-IDF + Linear SVM`：一对多线性支持向量机，合页损失

**神经网络（PyTorch）**

- `TextCNN`：多尺度一维卷积（卷积核 3/4/5，各 100 个）+ 全局最大池化
- `BiLSTM`：双向 LSTM + 加性注意力池化

---

## 主要结论

实验发现（详见论文第 5 节）：

1. **在中文任务上，线性模型与神经网络的差距很小甚至反超。**
   在 TNews 十分类任务上，TF-IDF + 逻辑回归的宏平均 F1 与 TextCNN 相当或更高。
2. **在效率上线性模型有压倒性优势。**
   线性模型训练时间通常在 0.1 秒量级，参数量约为神经网络的百分之一。
3. **神经网络的优势主要体现在文本较长的英文数据集上**（如 20 Newsgroups），
   此时词序信息对分类更有价值。

---

## 一个值得记录的实现细节

中文文本没有空格分隔词，因此 **TF-IDF 不能沿用 sklearn 默认的词切分规则**。
若直接使用默认的 `token_pattern`，一整句中文会被当作**一个** token，
导致特征维度极低（实验中仅为 287 维）且模型退化为随机猜测（宏平均 F1 ≈ 9%，随机基线 10%）。

本项目统一使用同一套分词函数（英文按空白切分、中文按字符切分）
作为 TF-IDF 的 `analyzer`，并自行拼接 1-gram 与 2-gram，
修正后 TNews 上的宏平均 F1 从 0.09 提升到 0.53。

这个坑本身也印证了参考论文的观点：
**如果没有一个调好的简单基线，很难发现流程中存在的严重问题。**

---

## 引用

```bibtex
@inproceedings{lin2023linear,
  title     = {Linear Classifier: An Often-Forgotten Baseline for Text Classification},
  author    = {Lin, Yu-Chen and Chen, Si-An and Liu, Jie-Jyun and Lin, Chih-Jen},
  booktitle = {Proceedings of ACL 2023 (Short Papers)},
  pages     = {1876--1888},
  year      = {2023}
}
```
