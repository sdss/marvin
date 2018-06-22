# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-06-21 15:11:40
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-06-21 17:18:21

from __future__ import print_function, division, absolute_import
import pkgutil
import importlib
import os


pkg_dir = os.path.dirname(__file__)
for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
    if name != 'base':
        importlib.import_module('marvin.contrib.vacs.{0}'.format(name), __package__)


from marvin.contrib.vacs.base import VACMixIn

