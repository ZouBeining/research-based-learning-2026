"""
数据加载与预处理模块

设计要点：
1. 数据来源：优先读取 dataset_downloader.py 缓存的真实数据集 (data/*.jsonl)
2. 特征表示：
   - 线性模型 (LR / SVM)：TF-IDF 稀疏特征
   - 神经网络 (TextCNN / BiLSTM)：词表索引 + 定长 padding 序列
3. 切分：按 训练/验证/测试 = 7:1:2 分层切分，固定随机种子保证可复现
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


# ==================== 数据结构 ====================

@dataclass
class Dataset:
    """数据集容器"""
    name: str
    language: str            # 'en' 或 'zh'
    texts: List[str]
    labels: List[int]
    label_names: List[str]
    source: str = ''
    cache: Dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    @property
    def n_samples(self) -> int:
        return len(self.texts)

    def stats(self) -> Dict:
        """数据集统计信息（用于论文表格）"""
        counts = Counter(self.labels)
        if self.language == 'en':
            lengths = [len(t.split()) for t in self.texts]
        else:
            lengths = [len(t.replace(' ', '')) for t in self.texts]
        return {
            'name': self.name,
            'language': self.language,
            'source': self.source,
            'n_samples': self.n_samples,
            'n_classes': self.n_classes,
            'label_distribution': {self.label_names[k]: counts[k]
                                   for k in sorted(counts)},
            'avg_len': float(np.mean(lengths)),
            'median_len': float(np.median(lengths)),
            'max_len': int(np.max(lengths)),
            'min_len': int(np.min(lengths)),
        }


# ==================== 文本清洗与分词 ====================

def clean_text(text: str, language: str = 'en') -> str:
    """基础文本清洗"""
    if language == 'en':
        text = text.lower()
        text = re.sub(r'\\[a-z]', ' ', text)          # 去除 20NG 中的转义字符
        text = re.sub(r'[^a-z0-9\s\.\,\!\?\;\:\'\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # 过滤过短文本
        return text
    else:
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def tokenize(text: str, language: str = 'en') -> List[str]:
    """
    分词：
    - 英文：按空白切分
    - 中文：按字符切分（避免引入 jieba 等外部依赖，高中生易于理解）
    """
    if language == 'en':
        return text.split()
    return [c for c in text if not c.isspace()]


# ==================== 词表构建（供神经网络使用） ====================

class Vocab:
    """简单词表：token -> index"""

    def __init__(self, token2idx: Dict[str, int]):
        self.token2idx = token2idx
        self.idx2token = {v: k for k, v in token2idx.items()}

    def __len__(self) -> int:
        return len(self.token2idx)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.token2idx[UNK_TOKEN]
        return [self.token2idx.get(t, unk) for t in tokens]

    @classmethod
    def build(cls, tokenized_texts: List[List[str]],
              min_freq: int = 2, max_size: int = 30000) -> 'Vocab':
        counter = Counter()
        for toks in tokenized_texts:
            counter.update(toks)

        token2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, freq in counter.most_common():
            if freq < min_freq:
                break
            if len(token2idx) >= max_size:
                break
            token2idx[token] = len(token2idx)
        return cls(token2idx)


def encode_and_pad(tokenized_texts: List[List[str]], vocab: Vocab,
                   max_len: int) -> np.ndarray:
    """编码为定长索引序列，超长截断、不足补 PAD"""
    arr = np.zeros((len(tokenized_texts), max_len), dtype=np.int64)
    for i, toks in enumerate(tokenized_texts):
        ids = vocab.encode(toks[:max_len])
        arr[i, :len(ids)] = ids
    return arr


# ==================== 数据集加载（读取本地缓存） ====================

def _load_cached(name: str) -> Tuple[List[str], List[int], List[str], Dict]:
    path = os.path.join(DATA_DIR, f'{name}.jsonl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'未找到数据集缓存 {path}\n'
            f'请先运行: python dataset_downloader.py --dataset {name}'
        )
    texts, labels, label_names = [], [], []
    meta: Dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if i == 0 and '_meta' in obj:
                meta = obj['_meta']
                label_names = obj.get('label_names', [])
                continue
            texts.append(obj['text'])
            labels.append(int(obj['label']))
    return texts, labels, label_names, meta


DATASET_SPECS = {
    'ag_news':       {'name': 'AG News',              'language': 'en'},
    '20news_4class': {'name': '20 Newsgroups',        'language': 'en'},
    'tnews':         {'name': 'TNews',                'language': 'zh'},
    'chnsenticorp':  {'name': 'ChnSentiCorp',         'language': 'zh'},
}


def load_dataset(key: str) -> Dataset:
    """按键名加载缓存的数据集"""
    if key not in DATASET_SPECS:
        raise KeyError(f'未知数据集: {key}, 可选: {list(DATASET_SPECS)}')
    spec = DATASET_SPECS[key]
    texts, labels, label_names, meta = _load_cached(key)

    language = spec['language']
    texts = [clean_text(t, language) for t in texts]

    # 过滤清洗后为空的样本，并同步标签
    kept_t, kept_y = [], []
    for t, y in zip(texts, labels):
        if len(tokenize(t, language)) >= 3:
            kept_t.append(t)
            kept_y.append(y)

    return Dataset(
        name=spec['name'],
        language=language,
        texts=kept_t,
        labels=kept_y,
        label_names=label_names,
        source=meta.get('source', '')
    )


def load_all_datasets(keys: Optional[List[str]] = None) -> Dict[str, Dataset]:
    """加载所有可用数据集"""
    keys = keys or list(DATASET_SPECS.keys())
    datasets = {}
    for k in keys:
        try:
            datasets[k] = load_dataset(k)
        except FileNotFoundError as e:
            print(f'  [跳过] {k}: {e}')
    return datasets


# ==================== 数据准备（切分 + 特征提取） ====================

@dataclass
class PreparedData:
    """单个数据集切分并向量化后的结果"""
    dataset: Dataset
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    # 线性模型特征（TF-IDF 稀疏矩阵）
    X_train_tfidf: object
    X_val_tfidf: object
    X_test_tfidf: object
    tfidf_vocab_size: int

    # 神经网络特征（索引序列）
    X_train_seq: np.ndarray
    X_val_seq: np.ndarray
    X_test_seq: np.ndarray
    nn_vocab_size: int

    # 原始文本（用于案例分析）
    text_train: List[str]
    text_test: List[str]


def make_ngram_analyzer(language: str, ngram_range: Tuple[int, int]):
    """
    构造一个可调用 analyzer，同时负责分词与 n-gram 拼接。

    之所以自己拼 n-gram 而不依赖 sklearn 的 ngram_range：
    一旦把 analyzer 设为可调用对象，sklearn 就不再生成 n-gram，
    因此这里显式地把 1-gram 与 2-gram 都产出。
    中文按字符切分，word bigram 即为「相邻二字」组合。
    """
    lo, hi = ngram_range

    def analyzer(text: str) -> List[str]:
        toks = tokenize(text, language)
        grams: List[str] = []
        for n in range(lo, hi + 1):
            if n == 1:
                grams.extend(toks)
            else:
                grams.extend(''.join(toks[i:i + n]) for i in range(len(toks) - n + 1))
        return grams

    return analyzer


def prepare_data(dataset: Dataset,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 tfidf_max_features: int = 20000,
                 tfidf_ngram_range: Tuple[int, int] = (1, 2),
                 tfidf_min_df: int = 2,
                 nn_vocab_size: int = 30000,
                 nn_min_freq: int = 2,
                 nn_max_len: Optional[int] = None) -> PreparedData:
    """
    切分数据并生成两种特征表示。
    切分顺序：先分出测试集，再从剩余部分分出验证集（分层抽样）。
    """
    texts, labels = dataset.texts, dataset.labels

    # 1) 训练+验证 / 测试
    X_tv, X_test, y_tv, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels,
        random_state=random_state
    )
    # 2) 训练 / 验证
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_ratio, stratify=y_tv,
        random_state=random_state
    )

    y_train = np.array(y_train, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    # ---------- TF-IDF 特征（线性模型） ----------
    # 关键：中文没有空格，必须使用与神经网络一致的分词函数作为 analyzer，
    # 否则整句会被当成一个 token，特征维度极低、模型退化为随机猜测。
    vectorizer = TfidfVectorizer(
        analyzer=make_ngram_analyzer(dataset.language, tfidf_ngram_range),
        max_features=tfidf_max_features,
        sublinear_tf=True,
        min_df=tfidf_min_df,
        max_df=0.95,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    # ---------- 索引序列特征（神经网络） ----------
    tok_train = [tokenize(t, dataset.language) for t in X_train]
    tok_val = [tokenize(t, dataset.language) for t in X_val]
    tok_test = [tokenize(t, dataset.language) for t in X_test]

    vocab = Vocab.build(tok_train, min_freq=nn_min_freq, max_size=nn_vocab_size)

    if nn_max_len is None:
        lens = [len(t) for t in tok_train]
        # 取 95 分位数作为序列长度，兼顾信息量与计算开销
        nn_max_len = int(np.percentile(lens, 95))
        nn_max_len = max(16, min(nn_max_len, 512))

    X_train_seq = encode_and_pad(tok_train, vocab, nn_max_len)
    X_val_seq = encode_and_pad(tok_val, vocab, nn_max_len)
    X_test_seq = encode_and_pad(tok_test, vocab, nn_max_len)

    return PreparedData(
        dataset=dataset,
        y_train=y_train, y_val=y_val, y_test=y_test,
        X_train_tfidf=X_train_tfidf, X_val_tfidf=X_val_tfidf, X_test_tfidf=X_test_tfidf,
        tfidf_vocab_size=len(vectorizer.vocabulary_),
        X_train_seq=X_train_seq, X_val_seq=X_val_seq, X_test_seq=X_test_seq,
        nn_vocab_size=len(vocab),
        text_train=list(X_train), text_test=list(X_test),
    )


if __name__ == '__main__':
    from dataset_downloader import ensure_datasets

    print('确保数据集缓存存在...')
    ensure_datasets(list(DATASET_SPECS.keys()))

    print('\n加载全部数据集...')
    datasets = load_all_datasets()
    for key, ds in datasets.items():
        s = ds.stats()
        print(f"\n[{key}] {s['name']} ({s['language']})")
        print(f"  样本数={s['n_samples']}  类别数={s['n_classes']}  "
              f"平均长度={s['avg_len']:.1f}  最大长度={s['max_len']}")
        print(f"  类别分布={s['label_distribution']}")

    if datasets:
        key = next(iter(datasets))
        print(f"\n准备数据 [{key}] ...")
        d = prepare_data(datasets[key])
        print(f"  TF-IDF 维度: {d.X_train_tfidf.shape} 词表={d.tfidf_vocab_size}")
        print(f"  NN 序列: {d.X_train_seq.shape} 词表={d.nn_vocab_size}")
        print(f"  训练/验证/测试 = {len(d.y_train)}/{len(d.y_val)}/{len(d.y_test)}")
        print('\n✅ 数据管道自检通过')
