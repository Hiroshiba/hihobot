from hihobot.data import load_doc2vec_model, make_janome_model, to_words, to_vec

import argparse
import glob
import re
from functools import partial
from itertools import starmap
from pathlib import Path
from typing import List

import numpy as np

from hihobot.config import create_from_json as create_config
from hihobot.dataset import _load_char, _load_text
from hihobot.generator import Generator
from hihobot.transoformer import Transformer
from hihobot.utility import save_arguments


def _extract_number(f):
    s = re.findall("\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
        model_dir: Path,
        iteration: int = None,
        prefix: str = 'main_',
):
    if iteration is None:
        paths = model_dir.glob(prefix + '*.npz')
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + '{}.npz'.format(iteration))
    return model_path


def generate(
        model_dir: Path,
        model_iteration: int,
        model_config: Path,
        char_path: Path,
        text_path: Path,
        doc2vec_model_path: Path,
        max_length: int,
        num_test: int,
        sampling_maximum: bool,
        output_dir: Path,
        gpu: int,
):
    output_dir.mkdir(exist_ok=True)

    output = output_dir / model_dir.name
    output.mkdir(exist_ok=True)

    save_arguments(arguments, output / 'arguments.json')

    config = create_config(model_config)

    chars = _load_char(char_path if char_path is not None else Path(config.dataset.char_path))
    transformer = Transformer(chars=chars)

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    generator = Generator(
        config,
        model_path,
        transformer=transformer,
        gpu=gpu,
    )
    print(f'Loaded generator "{model_path}"')

    load_doc2vec_model(doc2vec_model_path if doc2vec_model_path is not None else config.dataset.doc2vec_model_path)
    make_janome_model()

    if text_path is None:
        text_path = Path(config.dataset.text_path)

    texts = _load_text(text_path)

    np.random.RandomState(config.dataset.seed).shuffle(texts)
    texts = texts[:num_test]

    for i, text in enumerate(texts):
        words = to_words(text)
        vec = to_vec(words)

        out_text = generator.generate(
            vec=vec,
            max_length=max_length,
            sampling_maximum=sampling_maximum,
        )

        print('correct:', text)
        print('predict:', out_text)
        print('------------------------------------------------')

        (output / f'{i}.txt').open('w').write(out_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', type=Path)
    parser.add_argument('--model_iteration', '-mi', type=int)
    parser.add_argument('--model_config', '-mc', type=Path)
    parser.add_argument('--char_path', '-cp', type=Path)
    parser.add_argument('--text_path', '-tp', type=Path)
    parser.add_argument('--doc2vec_model_path', '-dmp', type=Path)
    parser.add_argument('--max_length', '-ml', type=int, default=32)
    parser.add_argument('--num_test', '-nt', type=int, default=50)
    parser.add_argument('--sampling_maximum', '-sm', action='store_true')
    parser.add_argument('--output_dir', '-o', type=Path, default='./output/')
    parser.add_argument('--gpu', type=int)
    arguments = parser.parse_args()

    generate(
        model_dir=arguments.model_dir,
        model_iteration=arguments.model_iteration,
        model_config=arguments.model_config,
        char_path=arguments.char_path,
        text_path=arguments.text_path,
        doc2vec_model_path=arguments.doc2vec_model_path,
        max_length=arguments.max_length,
        num_test=arguments.num_test,
        sampling_maximum=arguments.sampling_maximum,
        output_dir=arguments.output_dir,
        gpu=arguments.gpu,
    )
