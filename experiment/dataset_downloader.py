#!/usr/bin/env python3
"""
数据集下载器
通过 HuggingFace datasets-server 公开 REST API 下载真实数据集，并缓存为 JSONL。

缓存目录: ./data/
    - ag_news.jsonl        (英文, 4 类新闻)
    - tnews.jsonl          (中文, 多类新闻)
    - chnsenticorp.jsonl   (中文, 2 类情感)
    - 20news_4class.jsonl  (英文, 4 类新闻, 由 sklearn 下载后缓存)

用法:
    python dataset_downloader.py --all
    python dataset_downloader.py --dataset ag_news --size 8000
"""

import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
HF_ROWS_API = 'https://datasets-server.huggingface.co/rows'
PAGE_SIZE = 100  # datasets-server 单次请求上限


def _fetch_json(url: str, retries: int = 4, timeout: int = 40) -> Dict:
    """带重试的 JSON 请求"""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f'请求失败: {url}\n{last_err}')


def _fetch_rows(dataset: str, config: str, split: str,
                offset: int, length: int = PAGE_SIZE) -> Dict:
    """获取指定区间的一页数据"""
    url = (f'{HF_ROWS_API}?dataset={urllib.parse.quote(dataset, safe="")}'
           f'&config={config}&split={split}&offset={offset}&length={length}')
    return _fetch_json(url)


def sample_dataset(dataset: str,
                   config: str = 'default',
                   split: str = 'train',
                   size: int = 8000,
                   max_per_class: Optional[int] = None,
                   seed: int = 42,
                   verbose: bool = True) -> Tuple[List[str], List[int], List[str]]:
    """
    从 HF 数据集随机采样 size 条样本。

    先查询总行数，再在 [0, total-size] 范围内随机取连续的若干页，
    若启用 max_per_class 则按类别上限截断（保证类别均衡）。
    返回 (texts, labels, label_names)。
    """
    rng = random.Random(seed)

    # 查询数据集基本信息（含总行数与标签名）
    first = _fetch_rows(dataset, config, split, offset=0, length=1)
    total = first['num_rows_total']
    features = {f['name']: f['type'] for f in first['features']}
    label_names: List[str] = []
    if 'label' in features and isinstance(features['label'], dict) and 'names' in features['label']:
        label_names = [str(n) for n in features['label']['names']]

    target = min(size, total)
    n_pages = (target + PAGE_SIZE - 1) // PAGE_SIZE

    if verbose:
        print(f'  [{dataset}] 总行数={total}, 目标采样={target}, 需请求 {n_pages} 页')

    # 随机起始偏移，避免总是取头部数据
    max_start = max(1, total - n_pages * PAGE_SIZE)
    start = rng.randrange(0, max_start + 1)

    texts: List[str] = []
    labels: List[int] = []
    for p in range(n_pages):
        offset = start + p * PAGE_SIZE
        if offset >= total:
            break
        length = min(PAGE_SIZE, total - offset)
        page = _fetch_rows(dataset, config, split, offset=offset, length=length)
        for item in page['rows']:
            row = item['row']
            texts.append(str(row['text']))
            labels.append(int(row['label']))
        if verbose and (p + 1) % 10 == 0:
            print(f'    已下载 {len(texts)} 条...')

    # 类别均衡化
    if max_per_class is not None:
        buckets: Dict[int, List[int]] = {}
        for i, y in enumerate(labels):
            buckets.setdefault(y, []).append(i)
        kept: List[int] = []
        for y, idxs in sorted(buckets.items()):
            rng.shuffle(idxs)
            kept.extend(idxs[:max_per_class])
        rng.shuffle(kept)
        texts = [texts[i] for i in kept]
        labels = [labels[i] for i in kept]

    # 若不足目标数量，重复采样补齐（一般不会触发）
    if verbose:
        from collections import Counter
        dist = Counter(labels)
        print(f'  [{dataset}] 采样完成: {len(texts)} 条, 类别分布={dict(sorted(dist.items()))}')

    return texts, labels, label_names


def save_jsonl(path: str, texts: List[str], labels: List[int],
               label_names: List[str], meta: Optional[Dict] = None):
    """保存为 JSONL，第一行为元信息"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        meta = meta or {}
        meta.update({'n_samples': len(texts), 'n_classes': len(label_names)})
        f.write(json.dumps({'_meta': meta, 'label_names': label_names}, ensure_ascii=False) + '\n')
        for t, y in zip(texts, labels):
            f.write(json.dumps({'text': t, 'label': y}, ensure_ascii=False) + '\n')


def load_jsonl(path: str) -> Tuple[List[str], List[int], List[str]]:
    """读取缓存的 JSONL"""
    texts, labels = [], []
    label_names: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if i == 0 and '_meta' in obj:
                label_names = obj.get('label_names', [])
                continue
            texts.append(obj['text'])
            labels.append(obj['label'])
    return texts, labels, label_names


# ==================== 各数据集下载入口 ====================

def download_ag_news(size: int = 8000, max_per_class: int = 2000, seed: int = 42) -> str:
    """AG News: 英文 4 类新闻标题+摘要"""
    print('下载 AG News (英文, 4 类)...')
    texts, labels, names = sample_dataset(
        'fancyzhx/ag_news', config='default', split='train',
        size=size, max_per_class=max_per_class, seed=seed
    )
    path = os.path.join(DATA_DIR, 'ag_news.jsonl')
    save_jsonl(path, texts, labels, names,
               meta={'name': 'AG News', 'language': 'en', 'source': 'fancyzhx/ag_news'})
    return path


def download_tnews(size: int = 5000, max_per_class: int = 500, n_classes: int = 10,
                   fetch_rows: Optional[int] = None, seed: int = 42) -> str:
    """TNews: 中文多类新闻标题 (CLUE 基准)"""
    print('下载 TNews (中文, 多类)...')
    # TNews 训练集共 15 类，各类占比约 6.6%，需多抓一些才能凑够每类 max_per_class 条
    fetch_rows = fetch_rows or min(size * 2, 12000)
    texts_all, labels_all, names_all = sample_dataset(
        'C-MTEB/TNews-classification', config='default', split='train',
        size=fetch_rows, max_per_class=None, seed=seed, verbose=True
    )
    # 选取样本量最充足的 n_classes 个类别，保证均衡
    from collections import Counter
    dist = Counter(labels_all)
    chosen = [c for c, _ in dist.most_common(n_classes)]
    chosen_set = set(chosen)

    rng = random.Random(seed)
    buckets: Dict[int, List[int]] = {}
    for i, y in enumerate(labels_all):
        if y in chosen_set:
            buckets.setdefault(y, []).append(i)

    kept: List[int] = []
    for y in sorted(buckets):
        idxs = buckets[y]
        rng.shuffle(idxs)
        kept.extend(idxs[:max_per_class])
    rng.shuffle(kept)

    texts = [texts_all[i] for i in kept]
    labels = [labels_all[i] for i in kept]

    # 重映射标签为 0..K-1，并给出可读类别名
    original_labels = sorted(set(labels))
    remap = {old: new for new, old in enumerate(original_labels)}
    labels = [remap[y] for y in labels]
    # CLUE TNews 标签编号 -> 类别名
    clue_names = {
        '100': 'news_story', '101': 'news_culture', '102': 'news_entertainment',
        '103': 'news_sports', '104': 'news_finance', '106': 'news_house',
        '107': 'news_car', '108': 'news_edu', '109': 'news_tech',
        '110': 'news_military', '112': 'news_travel', '113': 'news_world',
        '114': 'news_stock', '115': 'news_game', '116': 'news_science',
    }
    label_names = []
    for old in original_labels:
        raw = names_all[old] if old < len(names_all) else str(old)
        label_names.append(clue_names.get(raw, f'class_{raw}'))

    print(f'  [TNews] 最终: {len(texts)} 条, 类别={label_names}')
    path = os.path.join(DATA_DIR, 'tnews.jsonl')
    save_jsonl(path, texts, labels, label_names,
               meta={'name': 'TNews', 'language': 'zh', 'source': 'C-MTEB/TNews-classification'})
    return path


def download_chnsenticorp(size: int = 8000, max_per_class: int = 4000, seed: int = 42) -> str:
    """ChnSentiCorp: 中文二分类情感分析"""
    print('下载 ChnSentiCorp (中文, 2 类情感)...')
    texts, labels, names = sample_dataset(
        'lansinuote/ChnSentiCorp', config='default', split='train',
        size=size, max_per_class=max_per_class, seed=seed
    )
    if not names:
        names = ['Negative', 'Positive']
    path = os.path.join(DATA_DIR, 'chnsenticorp.jsonl')
    save_jsonl(path, texts, labels, names,
               meta={'name': 'ChnSentiCorp', 'language': 'zh',
                     'source': 'lansinuote/ChnSentiCorp'})
    return path


def download_20news_4class(max_per_class: int = 600, seed: int = 42) -> str:
    """20 Newsgroups: 英文 4 类（使用 sklearn 内置下载，自动缓存）"""
    print('下载 20 Newsgroups (英文, 4 类)...')
    from sklearn.datasets import fetch_20newsgroups

    categories = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.guns']
    label_names = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.guns']

    data = fetch_20newsgroups(
        subset='all', categories=categories, shuffle=True,
        random_state=seed, remove=('headers', 'footers', 'quotes')
    )

    rng = random.Random(seed)
    buckets: Dict[int, List[int]] = {}
    for i, y in enumerate(data.target):
        buckets.setdefault(int(y), []).append(i)

    kept: List[int] = []
    for y in sorted(buckets):
        idxs = buckets[y]
        rng.shuffle(idxs)
        kept.extend(idxs[:max_per_class])
    rng.shuffle(kept)

    texts = [data.data[i] for i in kept]
    labels = [int(data.target[i]) for i in kept]

    print(f'  [20NG] 最终: {len(texts)} 条')
    path = os.path.join(DATA_DIR, '20news_4class.jsonl')
    save_jsonl(path, texts, labels, label_names,
               meta={'name': '20 Newsgroups (4-class)', 'language': 'en',
                     'source': 'sklearn.datasets.fetch_20newsgroups'})
    return path


# ==================== 统一入口 ====================

SPECS = {
    'ag_news': download_ag_news,
    'tnews': download_tnews,
    'chnsenticorp': download_chnsenticorp,
    '20news_4class': download_20news_4class,
}


def ensure_datasets(names: List[str], force: bool = False, **kwargs) -> Dict[str, str]:
    """确保数据集已缓存到本地，返回 name -> path"""
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {}
    for name in names:
        path = os.path.join(DATA_DIR, f'{name}.jsonl')
        if os.path.exists(path) and not force:
            n = sum(1 for _ in open(path, encoding='utf-8')) - 1
            print(f'  [缓存命中] {name}: {path} ({n} 条)')
            paths[name] = path
            continue
        fn = SPECS[name]
        # 20NG 不支持 size 参数
        if name == '20news_4class':
            paths[name] = fn()
        else:
            import inspect
            allowed = set(inspect.signature(fn).parameters)
            kw = {k: v for k, v in kwargs.items() if k in allowed}
            paths[name] = fn(**kw)
    return paths


def main():
    parser = argparse.ArgumentParser(description='数据集下载器')
    parser.add_argument('--all', action='store_true', help='下载全部数据集')
    parser.add_argument('--dataset', nargs='+', choices=list(SPECS.keys()))
    parser.add_argument('--size', type=int, default=8000, help='每个数据集的采样规模')
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    args = parser.parse_args()

    names = list(SPECS.keys()) if args.all or not args.dataset else args.dataset

    print(f'>>> 数据目录: {DATA_DIR}')
    t0 = time.time()
    paths = ensure_datasets(names, force=args.force, size=args.size)
    print(f'\n✅ 完成，用时 {time.time()-t0:.1f}s')
    for k, v in paths.items():
        print(f'   {k}: {v}')


if __name__ == '__main__':
    main()
