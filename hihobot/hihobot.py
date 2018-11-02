from pathlib import Path
import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np

from hihobot.config import create_from_json as create_config
from hihobot.data import load_doc2vec_model, make_janome_model, to_words, to_vec
from hihobot.dataset import _load_char, _load_text
from hihobot.generator import Generator
from hihobot.transoformer import Transformer
from hihobot.utility import save_arguments

from hihobot.generator import Generator


class Hihobot(object):
    def __init__(
            self,
            model_path: Path,
            model_config: Path,
            char_path: Path,
            doc2vec_model_path: Path,
            max_length: int,
            sampling_maximum: bool,
            gpu: Optional[int],
    ):
        self._max_length = max_length
        self._sampling_maximum = sampling_maximum

        config = create_config(model_config)

        chars = _load_char(char_path if char_path is not None else Path(config.dataset.char_path))
        transformer = Transformer(chars=chars)

        self._generator = Generator(
            config,
            model_path,
            transformer=transformer,
            gpu=gpu,
        )
        print(f'Loaded generator "{model_path}"')

        load_doc2vec_model(doc2vec_model_path if doc2vec_model_path is not None else config.dataset.doc2vec_model_path)
        make_janome_model()

    def text_to_vec(self, text: str):
        words = to_words(text)
        vec = to_vec(words)
        return vec

    def generate(self, vec: np.ndarray):
        out_text = self._generator.generate(
            vec=vec,
            max_length=self._max_length,
            sampling_maximum=self._sampling_maximum,
        )
        return out_text
