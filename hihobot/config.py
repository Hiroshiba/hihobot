import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Union

from hihobot.utility import JSONEncoder


class DatasetConfig(NamedTuple):
    char_path: str
    text_path: str
    doc2vec_model_path: str
    seed: int
    num_test: int


class NetworkConfig(NamedTuple):
    n_layers: int
    in_size: int
    hidden_size: int
    out_size: int
    dropout: float


class LossConfig(NamedTuple):
    pass


class TrainConfig(NamedTuple):
    batchsize: int
    gpu: List[int]
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    optimizer_gradient_clipping: float
    linear_shift: Dict[str, Any]


class ProjectConfig(NamedTuple):
    name: str
    tags: List[str]


class Config(NamedTuple):
    dataset: DatasetConfig
    network: NetworkConfig
    loss: LossConfig
    train: TrainConfig
    project: ProjectConfig

    def save_as_json(self, path):
        d = _namedtuple_to_dict(self)
        json.dump(d, open(path, 'w'), indent=2, sort_keys=True, cls=JSONEncoder)


def _namedtuple_to_dict(o: NamedTuple):
    return {
        k: v if not hasattr(v, '_asdict') else _namedtuple_to_dict(v)
        for k, v in o._asdict().items()
    }


def create_from_json(s: Union[str, Path]):
    d = json.load(open(s))
    backward_compatible(d)

    return Config(
        dataset=DatasetConfig(
            char_path=d['dataset']['char_path'],
            text_path=d['dataset']['text_path'],
            doc2vec_model_path=d['dataset']['doc2vec_model_path'],
            seed=d['dataset']['seed'],
            num_test=d['dataset']['num_test'],
        ),
        network=NetworkConfig(
            n_layers=d['network']['n_layers'],
            in_size=d['network']['in_size'],
            hidden_size=d['network']['hidden_size'],
            out_size=d['network']['out_size'],
            dropout=d['network']['dropout'],
        ),
        loss=LossConfig(
        ),
        train=TrainConfig(
            batchsize=d['train']['batchsize'],
            gpu=d['train']['gpu'],
            log_iteration=d['train']['log_iteration'],
            snapshot_iteration=d['train']['snapshot_iteration'],
            stop_iteration=d['train']['stop_iteration'],
            optimizer=d['train']['optimizer'],
            optimizer_gradient_clipping=d['train']['optimizer_gradient_clipping'],
            linear_shift=d['train']['linear_shift'],
        ),
        project=ProjectConfig(
            name=d['project']['name'],
            tags=d['project']['tags'],
        )
    )


def backward_compatible(d: Dict):
    if 'dropout' not in d['network']:
        d['network']['dropout'] = 0.2

    if 'linear_shift' not in d['train']:
        d['train']['linear_shift'] = None
