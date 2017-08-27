#Domains
from domains.hvac import HVAC
from domains.nav import NAVI_BILINEAR

#OPTIMIZER
from optimizer.hvac import HVACOptimizer
from optimizer.nav import NAVOptimizer

CONFIGURATION = {
    {
        'optimizer': HVACOptimizer,
        'domain': HVAC,
        'step': [12, 24, 48, 96],
        'batch': 100,
        'top': 10,
        'log': 'data/hvac/'
    },
    {
        'optimizer': NAVOptimizer,
        'domain': 'linear',
        'step': [30, 60, 120],
        'batch': 100,
        'top': 10,
        'log': 'data/nav/linear'
    },
    {
        'optimizer': NAVOptimizer,
        'domain': NAVI_BILINEAR,
        'step': [30, 60, 120],
        'batch': 100,
        'top': 10,
        'log': 'data/nav/bilinear'
    },
    {
        'optimizer': NAVOptimizer,
        'domain': 'linear',
        'step': [12, 24, 48, 96],
        'batch': 100,
        'top': 10,
        'log': 'data/hvac/'
    },


}