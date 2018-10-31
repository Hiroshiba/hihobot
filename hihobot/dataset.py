from functools import partial
import json
from functools import partial
from pathlib import Path
from typing import List, Dict, NamedTuple

import chainer
import ndjson
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from janome.tokenizer import Tokenizer

from hihobot.config import DatasetConfig
from hihobot.transoformer import Transformer


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
            doc2vec_model: Doc2Vec,
            janome_model: Tokenizer,
    ):
        self.texts = texts
        self.transformer = transformer
        self.doc2vec_model = doc2vec_model
        self.janome_model = janome_model

    def __len__(self):
        return len(self.texts)

    def get_example(self, i):
        text = self.texts[i]
        words = [t.surface for t in self.janome_model.tokenize(text)]
        vec = self.doc2vec_model.infer_vector(words)

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

    num_test = config.num_test
    trains = texts[num_test:]
    tests = texts[:num_test]
    evals = trains[:num_test]

    doc2vec_model = Doc2Vec.load(str(config.doc2vec_model_path))
    janome_model = Tokenizer()

    _Dataset = partial(
        CharIdsDataset,
        transformer=transformer,
        doc2vec_model=doc2vec_model,
        janome_model=janome_model,
    )
    return {
        'train': _Dataset(trains),
        'test': _Dataset(tests),
        'train_eval': _Dataset(evals),
    }
