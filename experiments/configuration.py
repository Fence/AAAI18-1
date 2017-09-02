# Domains
from domains.hvac import HVAC
from domains.nav import NAVI_BILINEAR, NAVI_NONLINEAR, NAVI_LINEAR
from domains.reservoir import RESERVOIR_NONLINEAR, RESERVOIR_LINEAR
# OPTIMIZER
from optimizer.hvac import HVACOptimizer
from optimizer.nav import NAVOptimizer
from optimizer.reservoir import ReservoirOptimizer


CONFIGURATIONS = [
    {
        'optimizer': NAVOptimizer,
        'domain': NAVI_BILINEAR,
        'step': [30, 60, 120],
        'batch': 100,
        'dimension': 2,
        'top': 10,
        'log': 'data/nav/bilinear',
        'epoch': 1000,
        'initial_mean': 0.0,
        'initial_std': 0.005,
    },
    {
        'optimizer': NAVOptimizer,
        'domain': NAVI_NONLINEAR,
        'step': [30, 60, 120],
        'batch': 100,
        'dimension': 2,
        'top': 10,
        'log': 'data/nav/nonlinear',
        'epoch': 1000,
        'initial_mean': 0.0,
        'initial_std': 0.005,
    },
    {
        'optimizer': NAVOptimizer,
        'domain': NAVI_LINEAR,
        'step': [30, 60, 120],
        'batch': 100,
        'dimension': 2,
        'top': 10,
        'log': 'data/nav/linear',
        'epoch': 1000,
        'initial_mean': 0.0,
        'initial_std': 0.005,
    },
    {
        'optimizer': ReservoirOptimizer,
        'domain': RESERVOIR_LINEAR,
        'step': [30, 60, 120],
        'batch': 100,
        'dimension': 20,
        'top': 10,
        'log': 'data/reservoir/linear',
        'epoch': 500,
        'initial_mean': 0.0,
        'initial_std': 0.5,
    },
    {
        'optimizer': ReservoirOptimizer,
        'domain': RESERVOIR_NONLINEAR,
        'step': [30, 60, 120],
        'batch': 100,
        'dimension': 20,
        'top': 10,
        'log': 'data/reservoir/nonlinear',
        'epoch': 500,
        'initial_mean': 0.0,
        'initial_std': 0.5,
    },
    {
        'optimizer': HVACOptimizer,
        'domain': HVAC,
        'step': [12, 24, 48, 96],
        'batch': 100,
        'dimension': 60,
        'top': 10,
        'log': 'data/hvac/nonlinear',
        'epoch': 2000,
        'initial_mean': 5.0,
        'initial_std': 1.0,
    },
]
