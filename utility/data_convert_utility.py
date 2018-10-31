from typing import List

from chainer.dataset import to_device

from hihobot.dataset import Data


def data_convert(
        batch: List[Data],
        device=None,
):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    return dict(
        input_array=[to_device(device, d.input_array) for d in batch],
        target_ids=[to_device(device, d.target_ids) for d in batch],
        vec=[to_device(device, d.vec) for d in batch],
    )
