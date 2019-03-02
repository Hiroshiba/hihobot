import argparse
import random
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import ndjson

from hihobot.vectorizer import Vectorizer


def _load_text(p: Path):
    ds: List[Dict[str, str]] = ndjson.load(p.open(encoding="utf8"))
    return [d['str'] for d in ds]


def analyze_dataset(
        dataset_text_path: Path,
        dataset_char_path: Path,
        doc2vec_model_path: Path,
        num_sample: int,
        show_num: int,
):
    vectorizer = Vectorizer(path_doc2vec_model=doc2vec_model_path)

    vocabulary = list(vectorizer._doc2vec_model.wv.vocab.keys())

    texts = _load_text(dataset_text_path)
    texts = random.sample(texts, num_sample)

    unknown_words = [
        word
        for text in texts
        for word in vectorizer.to_words(text)
        if word not in vocabulary
    ]

    counter = Counter(unknown_words)
    pprint(counter.most_common(show_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_text_path', type=Path)
    parser.add_argument('--dataset_char_path', type=Path)
    parser.add_argument('--doc2vec_model_path', type=Path)
    parser.add_argument('--num_sample', type=int, default=3000)
    parser.add_argument('--show_num', type=int, default=100)
    args = parser.parse_args()

    analyze_dataset(
        dataset_text_path=args.dataset_text_path,
        dataset_char_path=args.dataset_char_path,
        doc2vec_model_path=args.doc2vec_model_path,
        num_sample=args.num_sample,
        show_num=args.show_num,
    )
