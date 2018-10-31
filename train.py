import argparse
from copy import copy
from pathlib import Path
from typing import Any, Dict

from chainer import cuda, optimizer_hooks, optimizers, training
from chainer.iterators import SerialIterator
from chainer.training import extensions
from chainer.training.updaters import StandardUpdater
from tb_chainer import SummaryWriter

from hihobot.config import create_from_json
from hihobot.dataset import create as create_dataset
from hihobot.model import Model, create_predictor
from utility.chainer_extension_utility import TensorBoardReport
from utility.data_convert_utility import data_convert

parser = argparse.ArgumentParser()
parser.add_argument('config_json_path', type=Path)
parser.add_argument('output', type=Path)
arguments = parser.parse_args()

config = create_from_json(arguments.config_json_path)
arguments.output.mkdir(exist_ok=True)
config.save_as_json((arguments.output / 'config.json').absolute())

# model
predictor = create_predictor(config.network)
model = Model(loss_config=config.loss, predictor=predictor)

if config.train.gpu is not None:
    model.to_gpu(config.train.gpu)
    cuda.get_device_from_id(config.train.gpu).use()

# dataset
dataset = create_dataset(config.dataset)
train_iter = SerialIterator(dataset['train'], config.train.batchsize, repeat=True, shuffle=True)
test_iter = SerialIterator(dataset['test'], config.train.batchsize, repeat=False, shuffle=False)
train_eval_iter = SerialIterator(dataset['train_eval'], config.train.batchsize, repeat=False, shuffle=False)


# optimizer
def create_optimizer(model):
    cp: Dict[str, Any] = copy(config.train.optimizer)
    n = cp.pop('name').lower()

    if n == 'adam':
        optimizer = optimizers.Adam(**cp)
    elif n == 'sgd':
        optimizer = optimizers.SGD(**cp)
    else:
        raise ValueError(n)

    optimizer.setup(model)

    if config.train.optimizer_gradient_clipping is not None:
        optimizer.add_hook(optimizer_hooks.GradientClipping(config.train.optimizer_gradient_clipping))

    return optimizer


optimizer = create_optimizer(model)

# updater
updater = StandardUpdater(
    iterator=train_iter,
    optimizer=optimizer,
    converter=data_convert,
    device=config.train.gpu,
)

# trainer
trigger_log = (config.train.log_iteration, 'iteration')
trigger_snapshot = (config.train.snapshot_iteration, 'iteration')
trigger_stop = (config.train.stop_iteration, 'iteration') if config.train.stop_iteration is not None else None

trainer = training.Trainer(updater, stop_trigger=trigger_stop, out=arguments.output)
tb_writer = SummaryWriter(Path(arguments.output))

ext = extensions.Evaluator(test_iter, model, data_convert, device=config.train.gpu)
trainer.extend(ext, name='test', trigger=trigger_log)
ext = extensions.Evaluator(train_eval_iter, model, data_convert, device=config.train.gpu)
trainer.extend(ext, name='train', trigger=trigger_log)

ext = extensions.snapshot_object(predictor, filename='main_{.updater.iteration}.npz')
trainer.extend(ext, trigger=trigger_snapshot)

trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
trainer.extend(extensions.observe_lr(), trigger=trigger_log)
trainer.extend(extensions.LogReport(trigger=trigger_log))
trainer.extend(extensions.PrintReport(['main/loss', 'test/main/loss']), trigger=trigger_log)
trainer.extend(TensorBoardReport(writer=tb_writer), trigger=trigger_log)
if trigger_stop is not None:
    trainer.extend(extensions.ProgressBar(trigger_stop))

trainer.run()
