# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-12 18:00:11
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-20 18:20:59

from __future__ import print_function, division, absolute_import
from marvin.api.api import Interaction
from marvin import config
import pytest


auths = [None, 'token', 'netrc']


@pytest.fixture(params=auths)
def mint(request):
    base = 'https://lore.sdss.utah.edu/'
    url = '/marvin/api/general/getroutemap/'
    if request.param is None:
        pytest.skip("no auth should fail")
    ii = Interaction(url, auth=request.param, send=False, base=base)
    yield ii
    ii = None


class TestInteraction(object):

    def test_auth(self, mint):
        assert mint.authtype in auths
        if mint.authtype:
            assert mint.authtype == mint.session.auth.authtype

    def test_auth_fail(self, monkeypatch):
        monkeypatch.setattr(config, 'access', 'collab')
        base = 'https://lore.sdss.utah.edu/'
        url = '/marvin/api/general/getroutemap/'
        with pytest.raises(AssertionError, match='Must have an authorization type set for collab access to MPLs!'):
            Interaction(url, auth=None, send=False, base=base, params={'release': 'MPL-11'})


