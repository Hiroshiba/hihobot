import json
from functools import partial
from pathlib import Path
from typing import Dict, List, NamedTuple

import chainer
import ndjson
import numpy as np

from hihobot.config import DatasetConfig
from hihobot.transoformer import Transformer
from hihobot.vectorizer import Vectorizer


class Data(NamedTuple):
    input_array: np.ndarray  # shape: (length+1, num_id)
    target_ids: np.ndarray  # shape: (length+1, )
    vec: np.ndarray  # shape: (num_vec, )


def _load_char(p: Path) -> List[str]:
    return json.load(p.open(encoding="utf8"))


def _load_text(p: Path):
    ds: List[Dict[str, str]] = ndjson.load(p.open(encoding="utf8"))
    return [d['str'] for d in ds]


class CharIdsDataset(chainer.dataset.DatasetMixin):
    def __init__(
            self,
            texts: List[str],
            transformer: Transformer,
            vectorizer: Vectorizer,
    ):
        self.texts = texts
        self.transformer = transformer
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.texts)

    def get_example(self, i):
        text = self.texts[i]
        words = self.vectorizer.to_words(text)
        vec = self.vectorizer.to_vec(words)

        char_ids = [self.transformer.to_char_id(c) for word in words for c in word]

        target_ids = np.array(self.transformer.push_end_id(char_ids), dtype=np.int32)

        input_array = np.array([self.transformer.to_array(char_id) for char_id in char_ids])
        input_array = self.transformer.unshift_start_array(input_array)

        return Data(
            input_array=input_array,
            target_ids=target_ids,
            vec=vec,
        )


def create(config: DatasetConfig):
    texts = _load_text(Path(config.text_path))
    np.random.RandomState(config.seed).shuffle(texts)

    chars = _load_char(Path(config.char_path))
    transformer = Transformer(chars=chars)

    vectorizer = Vectorizer(path_doc2vec_model=config.doc2vec_model_path)

    num_test = config.num_test
    trains = texts[num_test:]
    tests = texts[:num_test]
    evals = trains[:num_test]

    _Dataset = partial(
        CharIdsDataset,
        transformer=transformer,
        vectorizer=vectorizer,
    )
    return {
        'train': _Dataset(trains),
        'test': _Dataset(tests),
        'train_eval': _Dataset(evals),
    }
