from pathlib import Path
from typing import List, Union

import chainer
import numpy as np
from chainer import cuda
import chainer.functions as F

from hihobot.config import Config
from hihobot.data import load_doc2vec_model, make_janome_model
from hihobot.model import create_predictor
from hihobot.transoformer import Transformer
from hihobot.dataset import _load_char


class Generator(object):
    def __init__(
            self,
            config: Config,
            model_path: Path,
            transformer: Transformer,
            gpu: int = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.gpu = gpu

        self.transformer = transformer
        self.num_id = self.transformer.get_num_id()

        self.predictor = predictor = create_predictor(config.network, train=False)
        chainer.serializers.load_npz(str(model_path), predictor)

        if self.gpu is not None:
            predictor.to_gpu(self.gpu)
            cuda.get_device_from_id(self.gpu).use()

        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

    def generate(
            self,
            vec: np.ndarray,  # shape: (num_vec, )
            max_length: int,
            sampling_maximum: bool,
    ):
        vec = self.predictor.xp.asarray(vec)

        hs = None
        cs = None
        x = self.predictor.xp.asarray(self.transformer.get_start_array())  # shape: (num_char, )

        text = ""
        for i in range(max_length):
            x = F.expand_dims(F.concat([x, vec], axis=0), axis=0)  # shape: (1, num_id+num_vec)

            hs, cs, x = self.predictor.forward_one(
                hs=hs,
                cs=cs,
                x=x,
            )

            char_id = int(self.sampling(x, maximum=sampling_maximum))
            if char_id == self.transformer.get_end_id():
                break

            x = self.predictor.xp.asarray(self.transformer.to_array(char_id))

            char = self.transformer.to_char(char_id)
            text += char

        return text

    def sampling(self, softmax_dist: chainer.Variable, maximum=True):
        """
        :param softmax_dist: shape: (batch, num_id)
        :return: shape: (batch, )
        """
        xp = self.predictor.xp

        if maximum:
            sampled = xp.argmax(softmax_dist.data, axis=1)
        else:
            prob_np = lambda x: x if isinstance(x, np.ndarray) else x.get()  # cupy don't have random.choice method

            prob_list = F.softmax(softmax_dist)
            sampled = xp.array([
                np.random.choice(np.arange(self.num_id), p=prob_np(prob))
                for prob in prob_list.data
            ])
        return sampled
