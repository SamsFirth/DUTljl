import os
import sys
from . import conv2d_operation
from . import conv3d_operation
from . import gemm_operation
if '-m' not in sys.argv:
    from . import generator
from . import library
from . import manifest
from . import rank_2k_operation
from . import rank_k_operation
from . import symm_operation
from . import trmm_operation
from .library import *
install_source_path = os.path.join(__path__[0], 'source')
if os.path.isdir(install_source_path):
    source_path = install_source_path
else:
    source_path = os.path.join(__path__[0], '../..')