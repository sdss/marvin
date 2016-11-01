# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-10-19 17:36:00
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2016-10-19 22:39:19
#
# This is the Marvin setup
#
from setuptools import setup
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
      keywords='marvin manga astronomy MaNGA',
      url='https://github.com/marvin-manga/marvin',
      packages=['marvin'],
      package_dir={'marvin': 'python/marvin'},
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
          'Topic :: Software Development :: User Interfaces',
      ],
      )


