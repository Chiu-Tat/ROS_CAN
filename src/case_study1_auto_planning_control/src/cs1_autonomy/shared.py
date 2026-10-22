"""Load the installed shared node classes without executing their entry points."""
from functools import lru_cache
import importlib.util
from cs1_experiment.core import PACKAGE as SHARED_PACKAGE

@lru_cache(maxsize=None)
def shared_node(name):
    path = SHARED_PACKAGE/'scripts'/name
    spec = importlib.util.spec_from_file_location('cs1_shared_'+path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
