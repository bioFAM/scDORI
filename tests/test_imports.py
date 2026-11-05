import os
import subprocess
import sys


def run_python(code):
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


def test_core_imports_do_not_load_optional_packages():
    run_python("""
import sys
import scdori
assert 'torch' not in sys.modules
assert scdori.pp.__name__ == 'scdori.pp'
assert scdori.pl.__name__ == 'scdori.pl'
from scdori import scDoRI, train_model_grn, train_scdori_phases, get_latent_topics
from scdori._core.models import scDoRI as direct
assert scDoRI is direct
assert 'scDoRI' in dir(scdori)
for name in ('scanpy', 'matplotlib', 'seaborn', 'tangermeme', 'gtfparse', 'IPython'):
    assert name not in sys.modules, name
try:
    scdori.missing_attribute
except AttributeError:
    pass
else:
    raise AssertionError('Unknown attribute must raise AttributeError')
""")


def test_missing_optional_package_has_install_hint():
    run_python("""
import importlib.abc
import sys
class BlockPlot(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'matplotlib':
            raise ModuleNotFoundError('Blocked for test', name=fullname)
sys.meta_path.insert(0, BlockPlot())
import scdori
try:
    scdori.plot_topic_activation_heatmap
except ModuleNotFoundError as exc:
    assert 'scdori[plot]' in str(exc), str(exc)
else:
    raise AssertionError('Plotting must report the missing optional dependency')
assert 'plot_topic_activation_heatmap' not in vars(scdori)
""")
