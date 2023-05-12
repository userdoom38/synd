import inspect
import pkgutil
import unittest
import sys
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s\t] %(message)s'
)

console_handler.setFormatter(formatter)
log.addHandler(console_handler)

from synd import set_rng_seed
set_rng_seed(12748718)

def load_tests(loader, suite, pattern):
    """ https://stackoverflow.com/questions/29713541/recursive-unittest-discover """
    for imp, modname, _ in pkgutil.walk_packages(__path__):
        mod = imp.find_module(modname).load_module(modname)
        for test in loader.loadTestsFromModule(mod):
            suite.addTests(test)

    return suite

