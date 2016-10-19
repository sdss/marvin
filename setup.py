# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-10-19 17:36:00
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2016-10-19 18:04:05
#
# This is the Marvin setup
#
from distutils.core import setup
import os
import sys

NAME = 'marvin'

VERSION = '0.2.0b1'

setup(name=NAME,
      version=VERSION,
      license='BSD3',
      description='Toolsuite for dealing with the MaNGA dataset',
      long_description=__doc__,
      author='The Marvin Developers',
      author_email='havok2063@hotmail.com',
      url='https://github.com/havok2063/manga-marvin/',
      packages=['marvin'],
      package_dir={'marvin': 'python/marvin'},
      )


