# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-06-21 15:11:40
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-17 23:16:44

from __future__ import absolute_import, division, print_function

import importlib
import os
import pkgutil

from marvin.contrib.vacs.base import VACMixIn  # noqa


pkg_dir = os.path.dirname(__file__)
for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
    if name != 'base':
        importlib.import_module('marvin.contrib.vacs.{0}'.format(name), __package__)
