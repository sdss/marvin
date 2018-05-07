# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-03-08 18:08:34
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-05-07 12:46:09

from __future__ import print_function, division, absolute_import
from marvin import config
from marvin.core.exceptions import MarvinError
from brain import bconfig
from brain.core.exceptions import BrainError, BrainUserWarning
import pytest
import os


@pytest.fixture()
def netrc(monkeypatch, tmpdir):
    config.access = 'public'
    tmpnet = tmpdir.mkdir('netrc').join('.netrc')
    monkeypatch.setattr(bconfig, '_netrc_path', str(tmpnet))
    yield tmpnet


@pytest.fixture()
def goodnet(netrc):
    netrc.write('')
    os.chmod(bconfig._netrc_path, 0o600)
    yield netrc


@pytest.fixture()
def bestnet(goodnet):
    goodnet.write(write('data.sdss.org'))
    goodnet.write(write('api.sdss.org'))
    config._check_netrc()
    config.access = 'collab'
    config.setRelease('MPL-6')
    yield goodnet


def write(host):
    netstr = 'machine {0}\n'.format(host)
    netstr += '    login test\n'
    netstr += '    password test\n'
    netstr += '\n'
    return netstr


class TestVars(object):
    ''' test getting/setting variables '''

    @pytest.mark.parametrize('var, toval',
                             [('mode', 'remote'), ('access', 'public')])
    def test_set(self, monkeypatch, var, toval):
        defval = config.__getattribute__(var)
        assert defval != toval
        monkeypatch.setattr(config, var, toval)
        newval = config.__getattribute__(var)
        assert newval == toval

    @pytest.mark.parametrize('var, toval',
                             [('mode', 'super'), ('access', 'always')])
    def test_set_wrong(self, var, toval):
        with pytest.raises(ValueError) as cm:
            config.__setattr__(var, toval)
        assert 'config.{0} must be'.format(var) in str(cm.value)


class TestAccess(object):

    def test_bad_access(self, netrc):
        assert config.access == 'public'
        with pytest.raises(BrainError) as cm:
            config.access = 'collab'
        assert 'No .netrc file found in your HOME directory!' in str(cm.value)
        assert config.access == 'public'

    def test_public_access(self, bestnet):
        assert config.access == 'collab'
        assert 'MPL-5' in config._allowed_releases
        assert 'DR14' in config._allowed_releases
        config.access = 'public'
        assert 'MPL-5' not in config._allowed_releases
        assert 'DR14' in config._allowed_releases

    def test_tree(self, bestnet):
        assert config.access == 'collab'
        assert 'mangawork' in os.environ['MANGA_SPECTRO_REDUX']
        assert 'MPL' in config.release

        config.access = 'public'
        assert 'sas/dr' in os.environ['MANGA_SPECTRO_REDUX']
        assert 'DR' in config.release

        config.access = 'collab'
        assert 'sas/dr' in os.environ['MANGA_SPECTRO_REDUX']
        assert 'DR' in config.release


class TestReleases(object):

    @pytest.mark.parametrize('release', [('dr15'), ('dr14'), ('mpl-5')])
    def test_tree(self, bestnet, release):
        assert config.access == 'collab'
        assert 'mangawork' in os.environ['MANGA_SPECTRO_REDUX']
        assert 'MPL' in config.release

        config.setRelease(release)
        if 'mpl' in release:
            assert 'mangawork' in os.environ['MANGA_SPECTRO_REDUX']
        else:
            assert release in os.environ['MANGA_SPECTRO_REDUX']

        assert config.release == release.upper()

    @pytest.mark.parametrize('release', [('dr15'), ('mpl-6')])
    def test_drpall(self, bestnet, release):
        assert 'mangawork' in config.drpall
        config.setRelease(release)
        if config.drpall:
            word = 'mangawork' if 'mpl' in release else release
            assert word in config.drpall

    def test_invalid_release(self):
        with pytest.raises(MarvinError) as cm:
            config.setRelease('MPL-1')
        assert 'trying to set an invalid release version.' in str(cm.value)

    def test_invalid_dr(self):
        with pytest.raises(AssertionError) as cm:
            config.setDR('MPL-5')
        assert 'Must specify a DRXX version!' in str(cm.value)

    def test_invalid_mpl(self):
        with pytest.raises(AssertionError) as cm:
            config.setMPL('DR15')
        assert 'Must specify an MPL-X version!' in str(cm.value)


class TestNetrc(object):
    ''' test the netrc access '''

    @pytest.mark.parametrize('host, msg',
                             [('data.sdss.org', 'api.sdss.org not found in netrc. You will not have remote access to SDSS data'),
                              ('api.sdss.org', 'data.sdss.org not found in netrc. You will not be able to download SDSS data')],
                              ids=['noapi', 'nodata'])
    def test_only_one_host(self, goodnet, host, msg):
        goodnet.write(write(host))
        with pytest.warns(BrainUserWarning) as cm:
            config._check_netrc()

        assert msg in str(cm[0].message)

    def test_good_netrc(self, bestnet):
        config._check_netrc()
        assert config.access == 'collab'


class TestConfig(object):

    def test_exists(self):
        assert config is not None

    def test_bad_login(self):
        config.access = 'public'
        with pytest.raises(AssertionError) as cm:
            config.login()
        assert 'You must have collaboration access to login.' in str(cm.value)


