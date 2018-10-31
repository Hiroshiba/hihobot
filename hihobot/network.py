from typing import List, Union

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class DeepLSTM(chainer.Chain):
    def __init__(self, n_layers: int, in_size: int, hidden_size: int, out_size: int, dropout=0.2):
        super().__init__()

        with self.init_scope():
            self.lstm = L.NStepLSTM(
                n_layers=n_layers,
                in_size=in_size,
                out_size=hidden_size,
                dropout=dropout,
            )
            self.post_linear = L.Linear(in_size=hidden_size, out_size=out_size)

    def __call__(self, xs: List[Union[chainer.Variable, np.ndarray]]):
        """
        :param xs:  # shape: List[(length, ?)]
        """
        _, _, hs = self.lstm(hx=None, cx=None, xs=xs)  # shape: List[(length, ?)]

        sections = np.cumsum([len(h) for h in hs])

        hs = F.concat(hs, axis=0)  # shape: (all_length, ?)
        hs = self.post_linear(hs)  # shape: (all_length, ?)

        hs = F.split_axis(hs, sections, axis=0)  # shape: List[(length, ?)]
        return hs
