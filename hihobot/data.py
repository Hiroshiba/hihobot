from pathlib import Path
from typing import Union, List

from gensim.models import Doc2Vec
from janome.tokenizer import Tokenizer

_doc2vec_model = None
_janome_model = None


def load_doc2vec_model(p: Union[str, Path]):
    global _doc2vec_model
    if _doc2vec_model is not None:
        return
    _doc2vec_model = Doc2Vec.load(str(p))


def to_vec(words: List[str]):
    return _doc2vec_model.infer_vector(words)


def make_janome_model():
    global _janome_model
    if _janome_model is not None:
        return
    _janome_model = Tokenizer()


def to_words(text: str):
    return [t.surface for t in _janome_model.tokenize(text)]
