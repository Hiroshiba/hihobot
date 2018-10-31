from typing import List

import chainer
import chainer.functions as F
from chainer import Chain

from hihobot.config import LossConfig, NetworkConfig
from hihobot.network import DeepLSTM


def create_predictor(config: NetworkConfig):
    predictor = DeepLSTM(
        n_layers=config.n_layers,
        in_size=config.in_size,
        hidden_size=config.hidden_size,
        out_size=config.out_size,
    )
    return predictor


class Model(Chain):
    def __init__(self, loss_config: LossConfig, predictor: DeepLSTM) -> None:
        super().__init__()
        self.loss_config = loss_config
        with self.init_scope():
            self.predictor = predictor

    def __call__(
            self,
            input_array: List[chainer.Variable],  # shape: List[(length+1, num_id)]
            target_ids: List[chainer.Variable],  # shape: List[(length+1, )]
            vec: List[chainer.Variable],  # shape: List[(num_vec, )]
    ):
        input = [
            F.concat([ia, F.repeat(F.expand_dims(v, axis=0), len(ia), axis=0)], axis=1)
            for ia, v in zip(input_array, vec)
        ]  # shape: List[(length+1, num_id+num_vec)]

        output = self.predictor(input)  # shape: List[(length+1, ?)]

        output = F.concat(output, axis=0)  # shape: (all_length, ?)
        target = F.concat(target_ids, axis=0)  # shape: (all_length, )

        loss = F.softmax_cross_entropy(output, target)

        chainer.report(dict(
            loss=loss,
        ), self)
        return loss
