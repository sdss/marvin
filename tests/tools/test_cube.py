#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Filename: test_cube.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-25 18:06:38


import os
import re

import numpy as np
import pytest
from astropy import wcs

import marvin
from marvin import config, marvindb
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from tests import marvin_test_if
from marvin.tools.cube import Cube


@pytest.fixture(autouse=True)
def skipbins(galaxy):
    if galaxy.bintype.name not in ['HYB10', 'NONE']:
        pytest.skip('Skipping all bins for Cube tests')
    if galaxy.template.name not in ['MILESHC-MASTARSSP']:
        pytest.skip('Skipping all templates for Cube tests')


class TestCube(object):

    def test_cube_loadfail(self):
        with pytest.raises(MarvinError) as cm:
            Cube()
        assert 'no inputs defined' in str(cm.value)

    def test_cube_load_from_local_file_by_filename_success(self, galaxy):
        cube = Cube(filename=galaxy.cubepath)
        assert cube is not None
        assert os.path.abspath(galaxy.cubepath) == cube.filename

    def test_cube_load_from_local_file_by_filename_fail(self):
        with pytest.raises(AssertionError):
            Cube(filename='not_a_filename.fits')

    @pytest.mark.parametrize('objtype', [('maps'), ('models')])
    def test_cube_wrong_file(self, galaxy, objtype):
        path = galaxy.mapspath if objtype == 'maps' else galaxy.modelpath
        with pytest.raises(MarvinError) as cm:
            Cube(filename=path)
        assert 'Trying to open a DAP file with Marvin Cube' in str(cm.value)

    @pytest.mark.uses_db
    def test_cube_load_from_local_database_success(self, galaxy):
        """Tests for Cube Load by Database."""
        cube = Cube(mangaid=galaxy.mangaid)
        assert cube is not None
        assert galaxy.mangaid == cube.mangaid
        assert galaxy.plate == cube.plate
        assert galaxy.dec == cube.dec
        assert galaxy.ra == cube.ra

    @pytest.mark.uses_db
    @pytest.mark.parametrize('plateifu, mode, errmsg',
                             [('8485-0923', 'local', 'Could not retrieve cube for '
                                                     'plate-ifu 8485-0923: No Results Found')],
                             ids=['noresults'])
    def test_cube_from_db_fail(self, plateifu, mode, errmsg):
        with pytest.raises(MarvinError) as cm:
            Cube(plateifu=plateifu, mode=mode)
        assert errmsg in str(cm.value)

    @marvin_test_if(mark='include', cube={'plateifu': '8485-1901',
                                          'release': 'MPL-6',
                                          'mode': 'local',
                                          'data_origin': 'file'})
    def test_quatities_reorder(self, cube):
        """Asserts the unit survives a quantity reorder (issue #374)."""

        flux = cube.flux
        spectral_resolution = cube.spectral_resolution

        assert flux.unit is not None
        assert spectral_resolution.unit is not None

        reordered_flux = np.moveaxis(flux, 0, -1)
        reordered_spectral_resolution = np.moveaxis(spectral_resolution, 0, -1)

        assert reordered_flux.unit is not None
        assert reordered_spectral_resolution.unit is not None

    @pytest.mark.uses_web
    @pytest.mark.parametrize('monkeyconfig',
                             [('release', 'DR15')],
                             ids=['dr15'], indirect=True)
    def test_cube_remote_drpver_differ_from_global(self, galaxy, monkeyconfig):

        if galaxy.release == 'DR15':
            pytest.skip('Skipping release for forced global DR15')

        drpver, dapver = config.lookUpVersions(config.release)
        assert config.release == 'DR15'
        cube = Cube(plateifu=galaxy.plateifu, mode='remote', release=galaxy.release)
        assert cube.release != config.release
        assert cube._drpver != drpver

    def test_cube_redshift(self, cube, galaxy):
        assert cube.data_origin == cube.exporigin
        redshift = cube.nsa.redshift \
            if cube.release == 'MPL-4' and cube.data_origin == 'file' else cube.nsa.z
        assert redshift == pytest.approx(galaxy.redshift)

    def test_release(self, cube, galaxy):
        #cube = Cube(plateifu=galaxy.plateifu)
        assert cube.release == galaxy.release

    def test_set_release_fails(self, cube):
        #cube = Cube(plateifu=galaxy.plateifu)
        with pytest.raises(MarvinError, match='the release cannot be changed'):
            cube.release = 'a'
        #assert 'the release cannot be changed' in str(ee.value)

    def test_load_filename_does_not_exist(self):
        """Tries to load a file that does not exist, in auto mode."""
        with pytest.raises(AssertionError) as ee:
            Cube(filename='hola.fits', mode='auto')

        assert re.match(r'filename .*hola.fits does not exist', str(ee.value)) is not None

    def test_load_filename_remote(self):
        """Tries to load a filename in remote mode and fails."""
        with pytest.raises(MarvinError) as ee:
            Cube(filename='hola.fits', mode='remote')

        assert 'filename not allowed in remote mode' in str(ee.value)

    @pytest.mark.uses_db
    def test_getFullPath_no_plateifu(self, galaxy):
        cube = Cube(mangaid=galaxy.mangaid)
        cube.plateifu = None
        assert cube._getFullPath() is None

    @pytest.mark.uses_db
    def test_download_no_plateifu(self, galaxy):
        cube = Cube(mangaid=galaxy.mangaid)
        cube.plateifu = None
        assert cube.download() is None

    def test_repr(self, cube):
        #cube = Cube(filename=galaxy.plateifu)
        args = cube.plateifu, cube.mode, cube.data_origin
        expected = "<Marvin Cube (plateifu='{0}', mode='{1}', data_origin='{2}')>".format(*args)
        assert cube.__repr__() == expected

    def test_load_cube_from_file_with_data(self, galaxy):
        cube = Cube(filename=galaxy.cubepath)
        cube._load_cube_from_file(data=cube.data)

    def test_load_cube_from_file_OSError(self, galaxy):
        cube = Cube(filename=galaxy.cubepath)
        cube.filename = 'hola.fits'
        with pytest.raises((IOError, OSError)) as ee:
            cube._load_cube_from_file()

        assert 'filename {0} cannot be found'.format(cube.filename) in str(ee.value)

    def test_load_cube_from_file_filever_ne_release(self, galaxy):
        release_wrong = 'DR15'
        with pytest.warns(MarvinUserWarning) as record:
            cube = Cube(filename=galaxy.cubepath, release=release_wrong)
        assert len(record) >= 1
        subrec = [r for r in record if 'mismatch' in r.message.args[0]]
        assert subrec[-1].message.args[0] == (
            'mismatch between file release={0} '.format(galaxy.release) +
            'and object release={0}. '.format(release_wrong) +
            'Setting object release to {0}'.format(galaxy.release))

        assert cube._release == galaxy.release

    def test_load_cube_from_db_disconnected(self, galaxy, monkeypatch):
        monkeypatch.setattr(marvindb, 'isdbconnected', False)
        with pytest.raises(MarvinError, match='No DB connected'):
            Cube(plateifu=galaxy.plateifu)

    @pytest.mark.uses_db
    def test_load_cube_from_db_data(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        cube._load_cube_from_db(data=cube.data)

    @marvin_test_if('include', data_origin=['db'])
    @pytest.mark.slow
    @pytest.mark.uses_db
    def test_getExtensionData_db(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        cube._get_extension_data('flux')

    def test_getrss(self, cube):

        rss = cube.getRSS()

        assert isinstance(rss, marvin.tools.RSS)
        assert cube.mode == rss.mode
        assert cube.plateifu == rss.plateifu
        assert cube.mangaid == rss.mangaid
        assert cube.release == rss.release

    @pytest.mark.parametrize('mode', [pytest.param('db', marks=pytest.mark.uses_db), ('file')])
    def test_get_available_bintypes(self, galaxy, mode):
        if mode == 'db':
            cube = Cube(plateifu=galaxy.plateifu)
        else:
            cube = Cube(filename=galaxy.cubepath)
        bintypes = cube.get_available_bintypes()
        expbins = galaxy.dap['stellar_sigma'].keys()
        assert set(bintypes) == set(expbins)


class TestWCS(object):

    def test_wcs(self, cube):
        assert cube.data_origin == cube.exporigin
        assert isinstance(cube.wcs, wcs.WCS)
        comp = cube.wcs.wcs.pc if cube.data_origin == 'api' else cube.wcs.wcs.cd
        assert comp[1, 1] == pytest.approx(0.000138889)


class TestPickling(object):

    def test_pickling_file(self, temp_scratch, galaxy):
        cube = Cube(filename=galaxy.cubepath)
        assert cube.data_origin == 'file'
        assert isinstance(cube, Cube)
        assert cube.data is not None

        assert not os.path.isfile(galaxy.cubepath[0:-7] + 'mpf')
        cube_file = temp_scratch / 'test_cube.mpf'
        cube.save(str(cube_file))
        assert cube_file.exists() is True
        assert cube.data is not None

        cube = None
        assert cube is None

        cube_restored = Cube.restore(str(cube_file))
        assert cube_restored.data_origin == 'file'
        assert isinstance(cube_restored, Cube)
        assert cube_restored.data is not None

    def test_pickling_file_custom_path(self, temp_scratch, galaxy):
        cube = Cube(filename=galaxy.cubepath)
        assert cube.data_origin == 'file'
        assert isinstance(cube, Cube)
        assert cube.data is not None

        test_path = temp_scratch / 'cubepickle' / 'test_cube.mpf'
        assert test_path.exists() is False

        path = cube.save(path=str(test_path))
        assert test_path.exists() is True
        assert path == os.path.realpath(os.path.expanduser(str(test_path)))

        cube_restored = Cube.restore(str(test_path), delete=True)
        assert cube_restored.data_origin == 'file'
        assert isinstance(cube_restored, Cube)
        assert cube_restored.data is not None

        assert not os.path.exists(path)

    @pytest.mark.uses_db
    def test_pickling_db(self, galaxy, temp_scratch):
        cube = Cube(plateifu=galaxy.plateifu)
        assert cube.data_origin == 'db'

        file = temp_scratch / 'test_cube_db.mpf'
        with pytest.raises(MarvinError) as cm:
            cube.save(str(file))

        assert 'objects with data_origin=\'db\' cannot be saved.' in str(cm.value)

    @pytest.mark.uses_web
    def test_pickling_api(self, temp_scratch, galaxy):
        cube = Cube(plateifu=galaxy.plateifu, mode='remote')
        assert cube.data_origin == 'api'
        assert isinstance(cube, Cube)
        assert cube.data is None

        test_path = temp_scratch / 'test_cube_api.mpf'

        cube.save(str(test_path))
        assert test_path.exists() is True

        cube = None
        assert cube is None

        cube_restored = Cube.restore(str(test_path))
        assert cube_restored.data_origin == 'api'
        assert isinstance(cube_restored, Cube)
        assert cube_restored.data is None


class TestMaskbit(object):

    @pytest.mark.parametrize('flag',
                             ['manga_target1',
                              'manga_target2',
                              'manga_target3',
                              'quality_flag',
                              'target_flags'])
    def test_flag(self, flag, cube):
        assert getattr(cube, flag, None) is not None
