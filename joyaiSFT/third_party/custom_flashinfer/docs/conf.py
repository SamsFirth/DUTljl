import os
import sys
from pathlib import Path
root = Path(__file__).parents[1].resolve()
sys.path.append(str(root))
os.environ['BUILD_DOC'] = '1'
autodoc_mock_imports = ['torch', 'triton', 'flashinfer.jit.aot_config', 'flashinfer._build_meta']
project = 'FlashInfer'
author = 'FlashInfer Contributors'
copyright = f'2023-2024, {author}'
package_version = (root / 'version.txt').read_text().strip()
version = package_version
release = package_version
extensions = ['sphinx_tabs.tabs', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.autosummary', 'sphinx.ext.mathjax']
autodoc_default_flags = ['members']
autosummary_generate = True
source_suffix = ['.rst']
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = 'furo'
templates_path = []
html_static_path = []
html_theme_options = {'logo_only': True}
html_static_path = ['_static']
html_theme_options = {'light_logo': 'FlashInfer-white-background.png', 'dark_logo': 'FlashInfer-black-background.png'}