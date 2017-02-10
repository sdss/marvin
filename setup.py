# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-10-19 17:36:00
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-09 13:08:40
#
# This is the Marvin setup
#

from setuptools import setup, find_packages
import os
import warnings
from get_version import generate_version_py


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def convert_md_to_rst(fp):
    try:
        import pypandoc
        output = pypandoc.convert_file(fp, 'rst')
        return output
    except ImportError:
        warnings.warn('cannot import pypandoc.', UserWarning)
        return open(fp).read()


data_files = []


def add_data_file(directory):
    extern_path = os.path.join(os.path.dirname(__file__), directory)
    for root, __, filenames in os.walk(extern_path):
        for filename in filenames:
            data_files.append(os.path.join('..', root.lstrip('python/'), filename))

add_data_file('python/marvin/extern/')
# add_data_file('python/marvin/web/configuration/')
# add_data_file('python/marvin/web/lib/')
# add_data_file('python/marvin/web/static/')
# add_data_file('python/marvin/web/templates/')
# add_data_file('python/marvin/web/uwsgi_conf_files/')
data_files.append('../marvin/db/dbconfig.ini')
data_files.append('../../requirements.txt')
data_files.append('../../README.md')


requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
install_requires = [line.strip().replace('==', '>=') for line in open(requirements_file)
                    if not line.strip().startswith('#') and line.strip() != '']

NAME = 'sdss-marvin'
VERSION = '2.1.0'
RELEASE = 'dev' not in VERSION
generate_version_py(NAME, VERSION, RELEASE)

setup(name=NAME,
      version=VERSION,
      license='BSD3',
      description='Toolsuite for dealing with the MaNGA dataset',
      long_description=convert_md_to_rst('README.md'),
      author='The Marvin Developers',
      author_email='havok2063@hotmail.com',
      keywords='marvin manga astronomy MaNGA',
      url='https://github.com/sdss/marvin',
      packages=find_packages(where='python', exclude=['marvin.web*']),
      package_dir={'': 'python'},
      package_data={'': data_files},
      install_requires=install_requires,
      scripts=['bin/run_marvin', 'bin/check_marvin'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Web Environment',
          'Framework :: Flask',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Topic :: Database :: Front-Ends',
          'Topic :: Documentation :: Sphinx',
          'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Software Development :: User Interfaces'
          #,
      ],
      )
