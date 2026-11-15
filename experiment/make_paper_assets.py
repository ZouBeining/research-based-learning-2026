#!/usr/bin/env python3
"""
论文素材生成器

读取 results/results.json，生成 LaTeX 表格与 pgfplots 图表。
与耗时的实验解耦，便于反复调整论文排版而不必重跑实验。

输出：
    results/tab_datasets.tex    数据集统计表
    results/tab_main.tex        主结果表（宏平均 F1）
    results/tab_efficiency.tex  效率表（训练时间 / 参数量）
    ../paper/figures/f1_comparison.tex   F1 分组柱状图
    ../paper/figures/train_time.tex      训练时间对数柱状图

用法：
    python make_paper_assets.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, 'results')
FIG_DIR = os.path.join(HERE, '..', 'paper', 'figures')

MODEL_ORDER = ['TF-IDF + Logistic Regression', 'TF-IDF + Linear SVM', 'TextCNN', 'BiLSTM']
MODEL_DISPLAY = {
    'TF-IDF + Logistic Regression': 'TF-IDF + LR',
    'TF-IDF + Linear SVM': 'TF-IDF + SVM',
    'TextCNN': 'TextCNN',
    'BiLSTM': 'BiLSTM',
}
LEGEND = ['线性：LR', '线性：SVM', '神经：TextCNN', '神经：BiLSTM']
# 灰度填充，保证黑白打印时四组柱子仍可区分
FILLS = ['black!12', 'black!35', 'black!60', 'black!85']
SHORT = {
    'ag_news': 'AG News',
    '20news_4class': '20 Newsgroups',
    'tnews': 'TNews',
    'chnsenticorp': 'ChnSentiCorp',
}
# 图表横轴用的短名（不含空格，避免 pgfplots 符号坐标解析问题）
SHORT_TIGHT = {
    'ag_news': 'AGNews',
    '20news_4class': '20NG',
    'tnews': 'TNews',
    'chnsenticorp': 'ChnSenti',
}
LANG_ZH = {'en': '英文', 'zh': '中文'}


def fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f'{n/1e6:.2f}M'
    if n >= 1_000:
        return f'{n/1e3:.1f}K'
    return str(n)


def fmt_time(t: float) -> str:
    if t >= 100:
        return f'{t:.0f}'
    if t >= 10:
        return f'{t:.1f}'
    return f'{t:.2f}'


def load():
    path = os.path.join(RESULT_DIR, 'results.json')
    if not os.path.exists(path):
        print(f'✗ 未找到 {path}，请先运行 python run_experiments.py')
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'  写入 {os.path.relpath(path, HERE)}')


def main():
    data = load()
    results = data['results']
    dstats = data['dataset_stats']
    fstats = data['feature_stats']

    # 只保留有结果的模型与数据集，保持稳定顺序
    models = [m for m in MODEL_ORDER if any(r['model'] == m for r in results)]
    keys = [k for k in dstats if any(r['dataset_key'] == k for r in results)]

    def find(key, model):
        for r in results:
            if r['dataset_key'] == key and r['model'] == model:
                return r
        return None

    print('\n生成论文素材...')

    # ---------------- 数据集统计表 ----------------
    lines = [
        '% 自动生成，请勿手工修改：数据集统计信息',
        '\\begin{table}[htbp]', '\\centering',
        '\\caption{实验数据集统计信息。长度指英文词数或中文字符数。}',
        '\\label{tab:datasets}', '\\small',
        '\\begin{tabular}{llccccc}', '\\toprule',
        '数据集 & 语言 & 训练 & 验证 & 测试 & 类别数 & 平均长度 \\\\',
        '\\midrule',
    ]
    for k in keys:
        s, f = dstats[k], fstats[k]
        lines.append(
            f"{SHORT.get(k, k)} & {LANG_ZH.get(s['language'], s['language'])} & "
            f"{f['n_train']} & {f['n_val']} & {f['n_test']} & "
            f"{s['n_classes']} & {s['avg_len']:.1f} \\\\"
        )
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    write(os.path.join(RESULT_DIR, 'tab_datasets.tex'), '\n'.join(lines))

    # ---------------- 主结果表 ----------------
    lines = [
        '% 自动生成，请勿手工修改：宏平均 F1',
        '\\begin{table}[htbp]', '\\centering',
        '\\caption{各模型在不同数据集上的宏平均 F1（\\%）。'
        '每组最高分以粗体标出。}',
        '\\label{tab:main-results}', '\\small',
        '\\begin{tabular}{l' + 'c' * len(keys) + '}', '\\toprule',
        '模型 & ' + ' & '.join(SHORT.get(k, k) for k in keys) + ' \\\\',
        '\\midrule',
    ]
    best = {k: max((find(k, m)['macro_f1'] for m in models if find(k, m)), default=0)
            for k in keys}
    for m in models:
        row = [MODEL_DISPLAY[m]]
        for k in keys:
            r = find(k, m)
            if r is None:
                row.append('--')
                continue
            v = r['macro_f1'] * 100
            s = f'{v:.2f}'
            if abs(v - best[k] * 100) < 1e-9:
                s = f'\\textbf{{{s}}}'
            row.append(s)
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    write(os.path.join(RESULT_DIR, 'tab_main.tex'), '\n'.join(lines))

    # ---------------- 效率表 ----------------
    lines = [
        '% 自动生成，请勿手工修改：训练时间与参数量',
        '\\begin{table}[htbp]', '\\centering',
        '\\caption{训练时间（秒）与可训练参数量。'
        '线性模型在该规模数据上的训练时间通常在 0.1 秒量级。}',
        '\\label{tab:efficiency}', '\\small',
        '\\begin{tabular}{ll' + 'c' * len(keys) + '}', '\\toprule',
        '模型 & 指标 & ' + ' & '.join(SHORT.get(k, k) for k in keys) + ' \\\\',
        '\\midrule',
    ]
    for m in models:
        row_t = [MODEL_DISPLAY[m], '训练时间']
        row_p = ['', '参数量']
        for k in keys:
            r = find(k, m)
            row_t.append(fmt_time(r['train_time_sec']) if r else '--')
            row_p.append(fmt_params(r['n_params']) if r else '--')
        lines.append(' & '.join(row_t) + ' \\\\')
        lines.append(' & '.join(row_p) + ' \\\\')
        if m != models[-1]:
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    write(os.path.join(RESULT_DIR, 'tab_efficiency.tex'), '\n'.join(lines))

    # ---------------- pgfplots: F1 柱状图 ----------------
    coords = ','.join(SHORT_TIGHT.get(k, k) for k in keys)
    lines = [
        '% 自动生成，请勿手工修改：宏平均 F1 对比',
        '\\begin{tikzpicture}',
        '\\begin{axis}[',
        '  ybar, bar width=8pt, width=\\linewidth, height=4.3cm,',
        '  ymin=0, ymax=105, ylabel={宏平均 F1 (\\%)},',
        f'  symbolic x coords={{{coords}}},',
        '  xtick=data, x tick label style={font=\\small},',
        '  legend style={at={(0.5,-0.20)}, anchor=north, legend columns=4, font=\\small, draw=none},',
        '  legend image code/.code={\draw[#1] (0cm,-0.12cm) rectangle (0.30cm,0.12cm);}',
        '  enlarge x limits=0.18, ymajorgrids, grid style={gray!25},',
        ']',
    ]
    for i, m in enumerate(models):
        pts = []
        for k in keys:
            r = find(k, m)
            pts.append(f'({SHORT_TIGHT.get(k, k)},{r["macro_f1"]*100:.2f})' if r
                       else f'({SHORT_TIGHT.get(k, k)},0)')
        lines.append(
            f'\\addplot+[fill={FILLS[i]}, draw=black] coordinates {{'
            + ' '.join(pts) + '};')
    lines += ['\\legend{' + ','.join(LEGEND[:len(models)]) + '}',
              '\\end{axis}', '\\end{tikzpicture}', '']
    write(os.path.join(FIG_DIR, 'f1_comparison.tex'), '\n'.join(lines))

    # ---------------- pgfplots: 训练时间柱状图 ----------------
    lines = [
        '% 自动生成，请勿手工修改：训练时间对比（对数坐标）',
        '\\begin{tikzpicture}',
        '\\begin{axis}[',
        '  ybar, bar width=8pt, width=\\linewidth, height=4.3cm,',
        '  ymode=log, ymin=0.02, ymax=200,',
        '  ylabel={训练时间 (秒，对数坐标)},',
        f'  symbolic x coords={{{coords}}},',
        '  xtick=data, x tick label style={font=\\small},',
        '  legend style={at={(0.5,-0.20)}, anchor=north, legend columns=4, font=\\small, draw=none},',
        '  legend image code/.code={\draw[#1] (0cm,-0.12cm) rectangle (0.30cm,0.12cm);}',
        '  enlarge x limits=0.18, ymajorgrids, grid style={gray!25},',
        ']',
    ]
    for i, m in enumerate(models):
        pts = []
        for k in keys:
            r = find(k, m)
            t = max(r['train_time_sec'], 0.02) if r else 0.02
            pts.append(f'({SHORT_TIGHT.get(k, k)},{t:.3f})')
        lines.append(
            f'\\addplot+[fill={FILLS[i]}, draw=black] coordinates {{'
            + ' '.join(pts) + '};')
    lines += ['\\legend{' + ','.join(LEGEND[:len(models)]) + '}',
              '\\end{axis}', '\\end{tikzpicture}', '']
    write(os.path.join(FIG_DIR, 'train_time.tex'), '\n'.join(lines))

    print('\n✅ 论文素材生成完成')


if __name__ == '__main__':
    main()
