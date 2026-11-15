#!/usr/bin/env python3
"""
主实验脚本：文本分类中线性分类器与神经网络的基线对比

流程：
    1. 加载真实数据集（AG News / 20 Newsgroups / TNews / ChnSentiCorp）
    2. 分层切分为 训练/验证/测试
    3. 提取两种特征：TF-IDF（线性模型）与词索引序列（神经网络）
    4. 训练并评估 4 个模型
    5. 输出 CSV / JSON / LaTeX 表格 / pgfplots 图表

用法：
    python run_experiments.py                # 完整实验
    python run_experiments.py --quick        # 快速模式（小数据、少轮次）
    python run_experiments.py --datasets ag_news tnews
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (  # noqa: E402
    DATASET_SPECS, PreparedData, load_all_datasets, prepare_data
)
from models import (  # noqa: E402
    LinearModelWrapper, ModelResult, TorchTrainer,
    create_bilstm, create_linear_svm, create_logistic_regression, create_textcnn,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'figures')

LINEAR_MODELS = ['TF-IDF + Logistic Regression', 'TF-IDF + Linear SVM']
NEURAL_MODELS = ['TextCNN', 'BiLSTM']
ALL_MODELS = LINEAR_MODELS + NEURAL_MODELS


# ==================== 配置 ====================

BASE_CONFIG: Dict[str, Any] = {
    # 数据
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    'tfidf_max_features': 20000,
    'tfidf_ngram_range': (1, 2),
    'tfidf_min_df': 2,
    'nn_vocab_size': 30000,
    'nn_min_freq': 2,

    # 线性模型
    'lr_C': 1.0,
    'lr_max_iter': 1000,
    'svm_C': 0.5,
    'svm_max_iter': 3000,

    # 神经网络
    'cnn_embed_dim': 128,
    'cnn_num_filters': 100,
    'cnn_filter_sizes': (3, 4, 5),
    'cnn_dropout': 0.5,
    'cnn_lr': 1e-3,
    'cnn_epochs': 8,
    'cnn_batch_size': 64,
    'cnn_patience': 3,

    'lstm_embed_dim': 128,
    'lstm_hidden_dim': 128,
    'lstm_num_layers': 1,
    'lstm_dropout': 0.5,
    'lstm_lr': 1e-3,
    'lstm_epochs': 8,
    'lstm_batch_size': 64,
    'lstm_patience': 3,
}

QUICK_CONFIG: Dict[str, Any] = {
    **BASE_CONFIG,
    'test_size': 0.2,
    'val_size': 0.1,
    'tfidf_max_features': 10000,
    'nn_vocab_size': 15000,
    'cnn_epochs': 3,
    'lstm_epochs': 3,
    'cnn_batch_size': 32,
    'lstm_batch_size': 32,
    'lr_max_iter': 300,
    'svm_max_iter': 1000,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_python(obj):
    """递归地把 numpy / torch 类型转成 Python 原生类型，便于 JSON 序列化"""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


# ==================== 实验运行器 ====================

class ExperimentRunner:

    def __init__(self, config: Dict[str, Any], output_dir: str = RESULT_DIR):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[ModelResult] = []
        self.dataset_stats: Dict[str, Dict] = {}
        self.feature_stats: Dict[str, Dict] = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(config['random_state'])

    # ---------- 单个模型 ----------

    def _run_linear(self, model_name: str, prep: PreparedData,
                    n_train: int) -> ModelResult:
        if model_name == 'TF-IDF + Logistic Regression':
            wrapper = create_logistic_regression(
                C=self.config['lr_C'], max_iter=self.config['lr_max_iter'],
                random_state=self.config['random_state']
            )
        else:
            wrapper = create_linear_svm(
                C=self.config['svm_C'], max_iter=self.config['svm_max_iter'],
                random_state=self.config['random_state'], calibrated=False
            )

        wrapper.fit(prep.X_train_tfidf, prep.y_train)
        res = wrapper.evaluate(prep.X_test_tfidf, prep.y_test,
                               prep.dataset.label_names)
        res.n_train = n_train
        return res

    def _run_neural(self, model_name: str, prep: PreparedData,
                    n_train: int) -> ModelResult:
        vocab_size = prep.nn_vocab_size
        n_classes = prep.dataset.n_classes

        if model_name == 'TextCNN':
            model, trainer = create_textcnn(
                vocab_size=vocab_size, num_classes=n_classes,
                embed_dim=self.config['cnn_embed_dim'],
                num_filters=self.config['cnn_num_filters'],
                filter_sizes=self.config['cnn_filter_sizes'],
                dropout=self.config['cnn_dropout'], device=self.device
            )
            trainer.optimizer = torch.optim.Adam(
                model.parameters(), lr=self.config['cnn_lr'], weight_decay=1e-4)
            epochs = self.config['cnn_epochs']
            batch_size = self.config['cnn_batch_size']
            patience = self.config['cnn_patience']
        else:
            model, trainer = create_bilstm(
                vocab_size=vocab_size, num_classes=n_classes,
                embed_dim=self.config['lstm_embed_dim'],
                hidden_dim=self.config['lstm_hidden_dim'],
                num_layers=self.config['lstm_num_layers'],
                dropout=self.config['lstm_dropout'],
                bidirectional=True, use_attention=True, device=self.device
            )
            trainer.optimizer = torch.optim.Adam(
                model.parameters(), lr=self.config['lstm_lr'], weight_decay=1e-4)
            epochs = self.config['lstm_epochs']
            batch_size = self.config['lstm_batch_size']
            patience = self.config['lstm_patience']

        # 类别不平衡时给损失函数加权
        counts = np.bincount(prep.y_train, minlength=n_classes).astype(np.float64)
        weights = len(prep.y_train) / (n_classes * np.maximum(counts, 1))
        trainer.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(self.device))

        trainer.fit(
            prep.X_train_seq, prep.y_train,
            prep.X_val_seq, prep.y_val,
            epochs=epochs, batch_size=batch_size,
            patience=patience, verbose=False
        )
        res = trainer.evaluate(prep.X_test_seq, prep.y_test,
                               prep.dataset.label_names)
        res.n_train = n_train
        return res

    def run_one(self, dataset_key: str, dataset, model_name: str,
                prep: PreparedData) -> Optional[ModelResult]:
        print(f'\n  ── {model_name}')
        t0 = time.time()
        try:
            if model_name in LINEAR_MODELS:
                res = self._run_linear(model_name, prep, len(prep.y_train))
            else:
                res = self._run_neural(model_name, prep, len(prep.y_train))
        except Exception as e:  # noqa: BLE001
            print(f'     ✗ 失败: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            return None

        res.dataset_name = dataset.name
        res.dataset_key = dataset_key
        res.model_name = model_name
        res.language = dataset.language
        print(f'     Acc={res.accuracy:.4f}  MacroF1={res.macro_f1:.4f}  '
              f'训练={res.train_time:.2f}s  推理={res.inference_time:.3f}s  '
              f'参数={res.n_params:,}  (总耗时 {time.time()-t0:.1f}s)')
        return res

    # ---------- 整个数据集 ----------

    def run_dataset(self, dataset_key: str, dataset,
                    model_names: List[str]) -> List[ModelResult]:
        print(f'\n{"="*70}')
        print(f'数据集: {dataset.name} [{dataset_key}]  语言={dataset.language}  '
              f'样本={dataset.n_samples}  类别={dataset.n_classes}')
        print(f'{"="*70}')

        prep = prepare_data(
            dataset,
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['random_state'],
            tfidf_max_features=self.config['tfidf_max_features'],
            tfidf_ngram_range=self.config['tfidf_ngram_range'],
            tfidf_min_df=self.config['tfidf_min_df'],
            nn_vocab_size=self.config['nn_vocab_size'],
            nn_min_freq=self.config['nn_min_freq'],
        )

        self.dataset_stats[dataset_key] = dataset.stats()
        self.feature_stats[dataset_key] = {
            'n_train': int(len(prep.y_train)),
            'n_val': int(len(prep.y_val)),
            'n_test': int(len(prep.y_test)),
            'tfidf_dim': int(prep.X_train_tfidf.shape[1]),
            'nn_vocab_size': int(prep.nn_vocab_size),
            'nn_seq_len': int(prep.X_train_seq.shape[1]),
        }
        print(f'切分: 训练={len(prep.y_train)} 验证={len(prep.y_val)} '
              f'测试={len(prep.y_test)}')
        print(f'特征: TF-IDF维度={prep.X_train_tfidf.shape[1]}  '
              f'NN词表={prep.nn_vocab_size}  序列长度={prep.X_train_seq.shape[1]}')

        results = []
        for name in model_names:
            r = self.run_one(dataset_key, dataset, name, prep)
            if r is not None:
                results.append(r)
        return results

    def run_all(self, datasets: Dict[str, Any],
                model_names: List[str]) -> List[ModelResult]:
        all_results = []
        for key, ds in datasets.items():
            all_results.extend(self.run_dataset(key, ds, model_names))
        self.results = all_results
        return all_results

    # ---------- 结果输出 ----------

    def save_csv(self) -> str:
        path = os.path.join(self.output_dir, 'results.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['dataset_key', 'dataset', 'language', 'model',
                        'accuracy', 'macro_f1', 'train_time_sec',
                        'inference_time_sec', 'n_params', 'n_train', 'n_test'])
            for r in self.results:
                fs = self.feature_stats.get(r.dataset_key, {})
                w.writerow([r.dataset_key, r.dataset_name, r.language, r.model_name,
                            f'{r.accuracy:.4f}', f'{r.macro_f1:.4f}',
                            f'{r.train_time:.4f}', f'{r.inference_time:.6f}',
                            r.n_params, fs.get('n_train', ''), fs.get('n_test', '')])
        print(f'\n结果 CSV -> {path}')
        return path

    def save_json(self) -> str:
        path = os.path.join(self.output_dir, 'results.json')
        payload = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'config': to_python(self.config),
            'dataset_stats': to_python(self.dataset_stats),
            'feature_stats': to_python(self.feature_stats),
            'results': to_python([{
                'dataset_key': r.dataset_key,
                'dataset': r.dataset_name,
                'language': r.language,
                'model': r.model_name,
                'accuracy': r.accuracy,
                'macro_f1': r.macro_f1,
                'train_time_sec': r.train_time,
                'inference_time_sec': r.inference_time,
                'n_params': r.n_params,
                'per_class_f1': r.per_class_f1,
            } for r in self.results]),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'结果 JSON -> {path}')
        return path

    def _find(self, key: str, model: str) -> Optional[ModelResult]:
        for r in self.results:
            if r.dataset_key == key and r.model_name == model:
                return r
        return None

    @staticmethod
    def _short_name(key: str) -> str:
        return {
            'ag_news': 'AG News',
            '20news_4class': '20NG',
            'tnews': 'TNews',
            'chnsenticorp': 'ChnSenti',
        }.get(key, key)

    @staticmethod
    def _fmt_params(n: int) -> str:
        if n >= 1_000_000:
            return f'{n/1e6:.2f}M'
        if n >= 1_000:
            return f'{n/1e3:.1f}K'
        return str(n)

    def print_summary(self):
        print(f'\n{"="*78}')
        print('实验结果汇总（宏平均 F1 %）')
        print(f'{"="*78}')
        keys = list(self.dataset_stats)
        hdr = f'{"模型":<32}' + ''.join(f'{self._short_name(k):>12}' for k in keys)
        print(hdr)
        print('-' * len(hdr))
        for m in ALL_MODELS:
            row = f'{m:<32}'
            for k in keys:
                r = self._find(k, m)
                row += f'{r.macro_f1*100:>12.2f}' if r else f'{"--":>12}'
            print(row)


def main():
    ap = argparse.ArgumentParser(description='文本分类基线对比实验')
    ap.add_argument('--quick', action='store_true', help='快速模式')
    ap.add_argument('--datasets', nargs='+', choices=list(DATASET_SPECS),
                    help='指定数据集')
    ap.add_argument('--models', nargs='+', choices=ALL_MODELS, help='指定模型')
    ap.add_argument('--output', default=RESULT_DIR, help='结果输出目录')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    config = dict(QUICK_CONFIG if args.quick else BASE_CONFIG)
    config['random_state'] = args.seed

    print('=' * 78)
    print('文本分类基线对比实验')
    print(f'模式: {"快速" if args.quick else "完整"}   设备: '
          f'{"cuda" if torch.cuda.is_available() else "cpu"}   '
          f'随机种子: {args.seed}')
    print('=' * 78)

    keys = args.datasets or list(DATASET_SPECS)
    print(f'\n加载数据集: {keys}')
    datasets = load_all_datasets(keys)
    if not datasets:
        print('\n✗ 没有可用数据集，请先运行: python dataset_downloader.py --all')
        sys.exit(1)

    model_names = args.models or ALL_MODELS

    runner = ExperimentRunner(config, output_dir=args.output)
    runner.run_all(datasets, model_names)

    if not runner.results:
        print('\n✗ 没有产生任何结果')
        sys.exit(1)

    runner.save_csv()
    runner.save_json()
    # 表格与图表由 make_paper_assets.py 从 results.json 生成，
    # 这样调整排版时无需重跑实验
    runner.print_summary()
    print('\n✅ 实验完成')


if __name__ == '__main__':
    main()
