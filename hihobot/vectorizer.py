from pathlib import Path
from typing import List, Union

from gensim.models import Doc2Vec
from janome.tokenizer import Tokenizer


class Vectorizer(object):
    def __init__(
            self,
            path_doc2vec_model: Union[str, Path]
    ) -> None:
        self._doc2vec_model = Doc2Vec.load(str(path_doc2vec_model))
        self._janome_model = Tokenizer()

    def to_vec(self, words: List[str]):
        return self._doc2vec_model.infer_vector(words)

    def to_words(self, text: str):
        return [t.surface for t in self._janome_model.tokenize(text)]
