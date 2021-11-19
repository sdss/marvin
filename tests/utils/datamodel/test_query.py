# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-11-15 10:27:30
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-16 10:12:52

from __future__ import print_function, division, absolute_import

import pytest
from marvin import config
from marvin.utils.datamodel.query import datamodel


PARAM_COUNT = {'MPL-4': {'all': 574, 'nospaxels': 309, 'nodap': 309},
               'MPL-5': {'all': 707, 'nospaxels': 322, 'nodap': 301},
               'MPL-6': {'all': 1863, 'nospaxels': 1008, 'nodap': 1031},
               'MPL-7': {'all': 1863, 'nospaxels': 1008, 'nodap': 1031},
               'DR15': {'all': 1863, 'nospaxels': 1008, 'nodap': 1031},
               'MPL-8': {'all': 2014, 'nospaxels': 1008, 'nodap': 1031},
               'DR16': {'all': 1863, 'nospaxels': 1008, 'nodap': 1031},
               'MPL-9': {'all': 2546, 'nospaxels': 1008, 'nodap': 1031}
               }


RELEASES = config._allowed_releases.keys()


@pytest.fixture(params=RELEASES)
def release(request):
    """Yield a release."""
    return request.param


@pytest.fixture
def paramtype():
    return 'all' if config._allow_DAP_queries else 'nodap'


class TestDataModel(object):


    def test_local_param_count(self, release, paramtype):

        dm = datamodel[release]
        assert len(dm.parameters) == PARAM_COUNT[release][paramtype]

    def test_remote_param_count(self, monkeypatch, db_off, release, paramtype):
        monkeypatch.setenv('MANGA_LOCALHOST', 0)
        dm = datamodel[release]
        assert len(dm.parameters) == PARAM_COUNT[release][paramtype]

