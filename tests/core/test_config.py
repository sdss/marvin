# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-03-08 18:08:34
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-20 18:12:02

from __future__ import print_function, division, absolute_import
import pytest
import os
import warnings

try:
    from urlparse import urlsplit, urlunsplit
except ImportError:
    from urllib.parse import urlsplit, urlunsplit

from marvin import config
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from brain import bconfig
from brain.core.exceptions import BrainError, BrainUserWarning


@pytest.fixture()
def setapi(monkeypatch):
    monkeypatch.setitem(bconfig._custom_config, 'use_test', False)
    bconfig._set_api_urls()


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
    config._check_access()
    config.access = 'collab'
    config.setRelease('DR17')
    yield goodnet


def write(host):
    netstr = 'machine {0}\n'.format(host)
    netstr += '    login test\n'
    netstr += '    password test\n'
    netstr += '\n'
    return netstr


@pytest.fixture()
def initconfig(monkeypatch):
    monkeypatch.delattr(config, '_tree')
    monkeypatch.setattr(config, '_release', None)


@pytest.fixture()
def set_default(monkeypatch, request):
    monkeypatch.setattr(config, '_release', request.param)
    monkeypatch.setitem(config._custom_config, 'default_release', request.param)
    config._check_config()


class TestVars(object):
    ''' test getting/setting variables '''

    @pytest.mark.parametrize('var, toval',
                             [('mode', 'remote'), ('access', 'collab')])
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
        assert 'DR15' in config._allowed_releases
        config.access = 'public'
        assert 'MPL-5' not in config._allowed_releases
        assert 'DR15' in config._allowed_releases

    def test_tree(self, bestnet):
        assert config.access == 'collab'
        assert 'dr17' in os.environ['MANGA_SPECTRO_REDUX']
        assert 'DR' in config.release

        config.access = 'public'
        assert 'sas/dr' in os.environ['MANGA_SPECTRO_REDUX']
        assert 'DR' in config.release

        config.access = 'collab'
        assert 'sas/dr' in os.environ['MANGA_SPECTRO_REDUX']
        assert 'DR' in config.release


class TestReleases(object):

    @pytest.mark.parametrize('release', [('dr15')])
    def test_tree(self, bestnet, release):
        assert config.access == 'collab'
        assert 'DR' in config.release

        config.setRelease(release)
        if 'mpl' in release:
            assert 'mangawork' in os.environ['MANGA_SPECTRO_REDUX']
        else:
            assert release in os.environ['MANGA_SPECTRO_REDUX']

        assert config.release == release.upper()

    @pytest.mark.parametrize('release', [('dr17')])
    def test_drpall(self, bestnet, release):
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
    def test_only_one_host(self, monkeypatch, goodnet, host, msg):
        monkeypatch.setitem(config._custom_config, 'check_access', True)
        goodnet.write(write(host))
        with pytest.warns(BrainUserWarning) as cm:
            config._check_access()

        assert msg in str(cm[0].message)

    def test_good_netrc(self, bestnet):
        config._check_access()
        assert config.access == 'collab'


class TestConfig(object):

    def test_exists(self):
        assert config is not None

    def test_bad_login(self):
        config.access = 'public'
        with pytest.raises(AssertionError) as cm:
            config.login()
        assert 'You must have collaboration access to login.' in str(cm.value)

    @pytest.mark.parametrize('defrel, exprel',
                             [('DR20', 'DR17'), ('bad_release', 'DR17')])
    def test_bad_default_release(self, initconfig, defrel, exprel):
        ''' this tests some initial conditions on config '''
        config._release = defrel
        config._check_config()
        msg = 'Release {0} is not in the allowed releases.  Switching to {1}'.format(defrel, exprel)
        with pytest.warns(MarvinUserWarning):
            warnings.warn(msg, MarvinUserWarning)
        assert config.release == exprel

    def test_summary_files(self):
        config.setDefaultDrpAll()  # need to reset the config
        drp, dap = config.lookUpVersions()
        assert config.drpall is not None
        assert config.dapall is not None
        assert 'drpall-{0}'.format(drp) in config.drpall
        assert 'dapall-{0}-{1}'.format(drp, dap) in config.dapall

    def test_summary_filepaths(self):
        config.setRelease("DR17")
        assert 'sas/dr17' in config.drpall
        assert 'sas/dr17' in config.dapall

    @pytest.mark.parametrize('name, vers, exp',
                             [('drpall', ('v3_1_1', None), 'sas/dr17/manga/spectro/redux/v3_1_1'),
                              ('dapall', ('v3_1_1', '3.1.0'), 'sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/')],
                             ids=['drpall', 'dapall'])
    def test_default_path(self, name, vers, exp):
        config.setRelease("DR17")
        drpver, dapver = vers
        path = config._get_default_path(name, drpver, dapver=dapver)
        assert exp in path

@pytest.mark.usefixtures('setapi')
class TestSasUrl(object):

    def test_sasurl_nonetrc(self, initconfig, netrc):
        assert 'DR' in config.release
        assert 'magrathea.sdss.org/' in config.sasurl

    @pytest.mark.parametrize('release',
                             [('DR17')],
                             ids=['public'])
    def test_sasurl(self, bestnet, release):
        assert 'magrathea.sdss.org' in config.sasurl
        config.setRelease(release)
        sasurl = 'magrathea.sdss.org' if 'DR' in release else 'magrathea.sdss.org'
        assert sasurl in config.sasurl

    @pytest.mark.parametrize('sas, exp',
                             [('utah', 'sas.sdss.org'),
                              ('public', 'dr17.sdss.org'),
                              ('test', 'lore.sdss.utah.edu'),
                              ('testpub', 'lore.sdss.utah.edu/public'),
                              ('local', 'localhost')],
                             )
    def test_sasurl_switch(self, sas, exp):
        public = sas == 'public'
        test = sas == 'test'
        if sas == 'testpub':
            public = test = True
        sas = 'utah' if sas != 'local' else sas
        config.switchSasUrl(sas, public=public, test=test)
        assert exp in config.sasurl

    @pytest.mark.parametrize('sas, exp',
                             [('utah', 'https://sas.sdss.org/marvin/api/cubes/8485-1901/'),
                              ('public', 'https://dr17.sdss.org/marvin/api/cubes/8485-1901/'),
                              ('test', 'https://lore.sdss.utah.edu/marvin/api/cubes/8485-1901/'),
                              ('testpub', 'https://lore.sdss.utah.edu/public/marvin/api/cubes/8485-1901/'),
                              ('local', 'http://localhost:5000/marvin/api/cubes/8485-1901/')],
                             )
    def test_sasurl_join(self, sas, exp):
        url = '/marvin/api/cubes/8485-1901/'
        public = sas == 'public'
        test = sas == 'test'
        if sas == 'testpub':
            public = test = True
        sas = 'utah' if sas != 'local' else sas
        config.switchSasUrl(sas, public=public, test=test)

        e = urlsplit(config.sasurl)
        t = urlsplit(url)
        final = urlunsplit(tuple(strjoin(*z) for z in zip(e, t)))
        assert exp == final


def strjoin(str1, str2):
    """ joins two url strings """
    if not str2.startswith(str1):
        return os.path.join(str1, str2.lstrip('/')) if str2 else str1
    else:
        return str2


# @pytest.mark.usefixtures('saslocal')
# class TestLogin(object):

#     def test_login_fail(self, monkeypatch):
#         monkeypatch.setattr(config, 'token', None)
#         assert config.token is None
#         with pytest.raises(AssertionError) as cm:
#             config.login()
#         assert 'You must have collaboration access to login.' in str(cm.value)

#     @pytest.mark.uses_web
#     def test_login(self, monkeypatch):
#         monkeypatch.setattr(config, 'token', None)
#         monkeypatch.setattr(config, 'access', 'collab')
#         assert config.token is None
#         config.login()
#         assert config.token is not None
