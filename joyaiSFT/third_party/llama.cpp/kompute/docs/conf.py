import sys
import os
import kp
import sphinx_material
project = 'Kompute'
copyright = '2020, The Institute for Ethical AI & Machine Learning'
html_title = 'Kompute Documentation (Python & C++)'
author = 'Alejandro Saucedo'
release = '0.8.1'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.githubpages', 'breathe', 'm2r2']
source_suffix = ['.rst', '.md']
breathe_default_project = 'Kompute'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_material'
if html_theme == 'sphinx_material':
    html_theme_options = {'google_analytics_account': 'G-F9LD9HL8LW', 'base_url': 'https://kompute.cc', 'color_primary': 'red', 'color_accent': 'light-blue', 'repo_url': 'https://github.com/KomputeProject/kompute/', 'repo_name': 'Kompute', 'globaltoc_depth': 2, 'globaltoc_collapse': False, 'globaltoc_includehidden': False, 'repo_type': 'github', 'nav_links': [{'href': 'https://github.com/KomputeProject/kompute/', 'internal': False, 'title': 'Kompute Repo'}]}
    extensions.append('sphinx_material')
    html_theme_path = sphinx_material.html_theme_path()
    html_context = sphinx_material.get_html_context()
html_sidebars = {'**': ['logo-text.html', 'globaltoc.html', 'localtoc.html', 'searchbox.html']}
html_static_path = ['_static']
html_css_files = ['assets/custom.css']