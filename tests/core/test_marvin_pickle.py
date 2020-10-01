# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-12 19:13:30
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-06-12 19:23:07

from __future__ import print_function, division, absolute_import
from marvin.core import marvin_pickle
import pytest
import os

data = dict(a=7, b=[10, 2])


class TestMarvinPickle(object):

    def test_specify_path(self, temp_scratch):
        file = temp_scratch.join('tmp_testMarvinPickleSpecifyPath.pck')
        path_out = marvin_pickle.save(obj=data, path=str(file), overwrite=False)
        assert file.check() is True
        revived_data = marvin_pickle.restore(path_out)
        assert data == revived_data

    def test_overwrite_true(self, temp_scratch):
        file = temp_scratch.join('tmp_testMarvinPickleOverwriteTrue.pck')
        open(str(file), 'a').close()
        path_out = marvin_pickle.save(obj=data, path=str(file), overwrite=True)
        assert file.check() is True
        revived_data = marvin_pickle.restore(path_out)
        assert data == revived_data

    def test_delete_on_restore(self, temp_scratch):
        file = temp_scratch.join('tmp_testMarvinPickleDeleteOnRestore.pck')
        path_out = marvin_pickle.save(obj=data, path=str(file), overwrite=False)
        assert file.check() is True
        revived_data = marvin_pickle.restore(path_out, delete=True)
        assert data == revived_data
        assert os.path.isfile(str(file)) is False


