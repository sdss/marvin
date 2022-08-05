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

    def _save_pickle(self, file, overwrite=False, delete=False):
        path_out = marvin_pickle.save(obj=data, path=str(file), overwrite=overwrite)
        assert file.exists() is True
        revived_data = marvin_pickle.restore(path_out, delete=delete)
        assert data == revived_data


    def test_specify_path(self, temp_scratch):
        file = temp_scratch / 'tmp_testMarvinPickleSpecifyPath.pck'
        self._save_pickle(file)

    def test_overwrite_true(self, temp_scratch):
        file = temp_scratch / 'tmp_testMarvinPickleOverwriteTrue.pck'
        open(str(file), 'a').close()
        self._save_pickle(file, overwrite=True)

    def test_delete_on_restore(self, temp_scratch):
        file = temp_scratch / 'tmp_testMarvinPickleDeleteOnRestore.pck'
        self._save_pickle(file, delete=True)
        assert os.path.isfile(str(file)) is False


