# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-12 19:00:10
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-06-12 19:13:15

from __future__ import print_function, division, absolute_import
from marvin.utils.general.structs import Dotable, DotableCaseInsensitive
import pytest

from collections import OrderedDict

normal_dicts = [OrderedDict(A=7, b=[10, 2], C='AbCdEf', d=['ghIJ', 'Lmnop'])]


@pytest.fixture(params=normal_dicts)
def dotdict(request):
    return Dotable(request.param)


@pytest.fixture(params=normal_dicts)
def dotdictci(request):
    return DotableCaseInsensitive(request.param)


class TestCoreDotable(object):

    @pytest.mark.parametrize('key', [('A'), ('b'), ('C'), ('d')])
    def test_dotable(self, dotdict, key):
        assert dotdict[key] == dotdict.__getattr__(key)

    @pytest.mark.parametrize('key', [('A'), ('b'), ('C'), ('d')])
    def test_dotablecaseins(self, dotdictci, key):
        assert dotdictci[key] == dotdictci.__getattr__(key)
        assert dotdictci[key.upper()] == dotdictci.__getattr__(key.lower())
        assert dotdictci[key.lower()] == dotdictci.__getattr__(key.upper())
        assert dotdictci[key.lower()] == dotdictci.__getattr__(key.lower())
