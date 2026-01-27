import pathlib
from pathlib import Path
import subprocess
import os
import copy
all_files = []
for p in sorted(Path('./').rglob('*')):
    if p.suffix in ['.hpp', '.cpp']:
        all_files.append(pathlib.PurePath(p))
for x in all_files:
    subprocess.Popen(f'dos2unix {str(x)}', shell=True)
    cmd = f'clang-format-12 -style=file -i {str(x)}'
    subprocess.Popen(cmd, shell=True)