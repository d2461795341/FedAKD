from alg.fedavg import fedavg
from alg.fedprox import fedprox
from alg.fedbn import fedbn
from alg.base import base
from alg.fedap import fedap
from alg.metafed import metafed
from alg.fedsoft import fedsoft
from alg.myfed import myfed
from alg.myfedmode import myfedmode
from alg.fedprox_FedAKD import fedprox_FedAKD

ALGORITHMS = [
    'fedavg',
    'fedprox',
    'fedbn',
    'base',
    'fedap',
    'metafed',
    'fedsoft',
    'myfed'
    'myfedmode'
    'fedprox_FedAKD'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
