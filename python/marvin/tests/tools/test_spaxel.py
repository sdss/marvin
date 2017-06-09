#!/usr/bin/env python
# encoding: utf-8
#
# test_spaxels.py
#
# Created by José Sánchez-Gallego on Sep 9, 2016.


from __future__ import division, print_function, absolute_import

import os

import pytest
import astropy.io.fits

from marvin import config
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.core.exceptions import MarvinError
from marvin.tools.analysis_props import DictOfProperties
from marvin.tools.spaxel import Spaxel
from marvin.tools.spectrum import Spectrum


# class TestSpaxelBase(marvin.tests.MarvinTest):
#     """Defines the files and plateifus we will use in the tests."""
# 
#     @classmethod
#     def setUpClass(cls):
# 
#         super(TestSpaxelBase, cls).setUpClass()
#         cls.set_sasurl('local')
#         cls._update_release('MPL-4')
#         cls.set_filepaths()
# 
#         cls.filename_cube = os.path.realpath(cls.cubepath)
#         cls.filename_maps_default = os.path.join(
#             cls.mangaanalysis, cls.drpver, cls.dapver,
#             'default', str(cls.plate), 'mangadap-{0}-default.fits.gz'.format(cls.plateifu))
# 
#     @classmethod
#     def tearDownClass(cls):
#         pass
# 
#     def setUp(self):
#         self._update_release('MPL-4')
#         config.forceDbOn()
# 
#         assert os.path.exists(self.filename_cube)
#         assert os.path.exists(self.filename_maps_default)
# 
#     def tearDown(self):
#         pass
# 

class TestSpaxelInit(object):
    
    def _spaxel_init(self, spaxel, cube, maps, spectrum):
        
        args = {'cube': cube, 'maps': maps, 'spectrum': spectrum}
        objs = {'cube': Cube, 'maps': Maps, 'spectrum': Spectrum}

        for key in objs:
            if args[key]:
                assert isinstance(getattr(spaxel, key), objs[key])
            else:
                assert getattr(spaxel, key) is None

        assert len(spaxel.properties) > 0

    def test_no_cube_no_maps_db(self, galaxy):
        spaxel = Spaxel(x=15, y=16, plateifu=galaxy.plateifu)
        self._spaxel_init(spaxel, cube=True, maps=True, spectrum=True)
        # assert isinstance(spaxel.cube, Cube)
        # assert isinstance(spaxel.maps, Maps)
        # assert isinstance(spaxel.spectrum, Spectrum)
        # assert len(spaxel.properties) > 0

    def test_cube_false_no_maps_db(self, galaxy):
        spaxel = Spaxel(x=15, y=16, plateifu=galaxy.plateifu, cube=False)
        self._spaxel_init(spaxel, cube=False, maps=True, spectrum=False)
        # assert spaxel.cube is None
        # assert isinstance(spaxel.maps, Maps)
        # assert spaxel.spectrum is None
        # assert len(spaxel.properties) > 0

    def test_no_cube_maps_false_db(self, galaxy):
        spaxel = Spaxel(x=15, y=16, plateifu=galaxy.plateifu, maps=False)
        assert spaxel.maps is None
        assert isinstance(spaxel.cube, Cube)
        assert isinstance(spaxel.spectrum, Spectrum)
        assert len(spaxel.properties) == 0

    def test_cube_object_db(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube)
        assert isinstance(spaxel.cube, Cube)
        assert isinstance(spaxel.maps, Maps)
        assert isinstance(spaxel.spectrum, Spectrum)
        assert len(spaxel.properties) > 0

    def test_cube_object_maps_false_db(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=False)
        assert isinstance(spaxel.cube, Cube)
        assert spaxel.maps is None
        assert isinstance(spaxel.spectrum, Spectrum)
        assert len(spaxel.properties) == 0

    def test_cube_maps_object_filename(self):

        cube = Cube(filename=self.filename_cube)
        maps = Maps(filename=self.filename_maps_default)
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=maps)

        assert isinstance(spaxel.cube, Cube)
        assert isinstance(spaxel.maps, Maps)

        assert isinstance(spaxel.spectrum, Spectrum)
        assert len(spaxel.properties) > 0

    def test_cube_maps_object_filename_mpl5(self):

        config.setMPL('MPL-5')

        cube = Cube(filename=self.filename_cube)
        maps = Maps(filename=self.filename_maps_default)
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=maps)

        assert cube._drpver == 'v1_5_1'
        assert spaxel._drpver == 'v1_5_1'
        assert maps._drpver == 'v1_5_1'
        assert maps._dapver == '1.1.1'
        assert spaxel._dapver == '1.1.1'

        assert isinstance(spaxel.cube, Cube)
        assert isinstance(spaxel.maps, Maps)

        assert isinstance(spaxel.spectrum, Spectrum)
        assert len(spaxel.properties) > 0

    def test_cube_object_api(self):

        cube = Cube(plateifu=self.plateifu, mode='remote')
        spaxel = Spaxel(x=15, y=16, cube=cube)

        assert isinstance(spaxel.cube, Cube)
        assert isinstance(spaxel.maps, Maps)

        assert isinstance(spaxel.spectrum, Spectrum)
        assert len(spaxel.properties) > 0

    def test_cube_maps_object_api(self):

        cube = Cube(plateifu=self.plateifu, mode='remote')
        maps = Maps(plateifu=self.plateifu, mode='remote')
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=maps)

        assert isinstance(spaxel.cube, Cube)
        assert isinstance(spaxel.maps, Maps)

        assert isinstance(spaxel.spectrum, Spectrum)
        assert len(spaxel.properties) > 0
        assert isinstance(spaxel.properties, DictOfProperties)

    def test_db_maps_miles(self):

        spaxel = Spaxel(x=15, y=16, cube=False, modelcube=False, maps=True,
                        plateifu=self.plateifu,
                        template_kin='MILES-THIN')
        assert spaxel.maps.template_kin == 'MILES-THIN'

    def test_api_maps_invalid_template(self):

        with pytest.raises(AssertionError) as cm:
            Spaxel(x=15, y=16, cube=False, modelcube=False, maps=True,
                   plateifu=self.plateifu,
                   template_kin='MILES-TH')
        assert 'invalid template_kin' in str(cm.exception)

    def test_load_false(self):

        spaxel = Spaxel(plateifu=self.plateifu, x=15, y=16, load=False)

        assert not spaxel.loaded
        assert spaxel.cube
        assert spaxel.maps
        assert spaxel.modelcube
        assert len(spaxel.properties) == 0
        assert spaxel.spectrum is None

        spaxel.load()

        assert isinstance(spaxel.cube, Cube)
        assert isinstance(spaxel.maps, Maps)

        assert isinstance(spaxel.spectrum, Spectrum)
        assert len(spaxel.properties) > 0
        assert isinstance(spaxel.properties, DictOfProperties)

    def test_fails_unbinned_maps(self):

        maps = Maps(plateifu=self.plateifu, bintype='VOR10',
                                      release='MPL-5')

        with pytest.raises(MarvinError) as cm:
            Spaxel(x=15, y=16, plateifu=self.plateifu, maps=maps)

        assert 'cannot instantiate a Spaxel from a binned Maps.' in str(cm.exception)

    def test_spaxel_ra_dec(self):

        cube = Cube(plateifu=self.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube)

        assert round(abs(spaxel.ra-232.54512), 5) == 0
        assert round(abs(spaxel.dec-48.690062), 5) == 0

    def test_release(self):

        cube = Cube(plateifu=self.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube)

        assert spaxel.release == 'MPL-4'

        with pytest.raises(MarvinError) as ee:
            spaxel.release = 'a'
            assert 'the release cannot be changed' in str(ee.exception)


class TestPickling(object):

    def setUp(self):
        super(TestPickling, self).setUp()
        self._files_created = []

    def tearDown(self):

        super(TestPickling, self).tearDown()

        for fp in self._files_created:
            full_fp = os.path.realpath(os.path.expanduser(fp))
            if os.path.exists(full_fp):
                os.remove(full_fp)

    def test_pickling_db_fails(self):

        cube = Cube(plateifu=self.plateifu)
        spaxel = cube.getSpaxel(x=1, y=3)

        spaxel_path = '~/test_spaxel.mpf'
        self._files_created.append(spaxel_path)
        with pytest.raises(MarvinError) as ee:
            spaxel.save(spaxel_path, overwrite=True)

        assert 'objects with data_origin=\'db\' cannot be saved.' in str(ee.exception)

    def test_pickling_only_cube_file(self):

        cube = Cube(filename=self.filename_cube)
        maps = Maps(filename=self.filename_maps_default)

        spaxel = cube.getSpaxel(x=1, y=3, properties=maps, modelcube=False)

        spaxel_path = '~/test_spaxel.mpf'
        self._files_created.append(spaxel_path)

        path_saved = spaxel.save(spaxel_path, overwrite=True)
        assert os.path.exists(path_saved)
        assert os.path.realpath(os.path.expanduser(spaxel_path)), path_saved

        del spaxel

        spaxel_restored = marvin.tools.spaxel.Spaxel.restore(spaxel_path)
        assert spaxel_restored is not None
        assert isinstance(spaxel_restored, marvin.tools.spaxel.Spaxel)

        assert spaxel_restored.cube is not None
        assert spaxel_restored.cube.data_origin == 'file'
        assert isinstance(spaxel_restored.cube.data, astropy.io.fits.HDUList)

        assert spaxel_restored.maps is not None
        assert spaxel_restored.maps.data_origin == 'file'
        assert isinstance(spaxel_restored.maps.data, astropy.io.fits.HDUList)

    def test_pickling_all_api(self):

        self._update_release('MPL-5')

        cube = Cube(plateifu=self.plateifu, mode='remote')
        maps = Maps(plateifu=self.plateifu, mode='remote')
        modelcube = marvin.tools.modelcube.ModelCube(plateifu=self.plateifu, mode='remote')

        spaxel = cube.getSpaxel(x=1, y=3, properties=maps, modelcube=modelcube)

        assert spaxel.cube.data_origin == 'api'
        assert spaxel.maps.data_origin == 'api'
        assert spaxel.modelcube.data_origin == 'api'

        spaxel_path = '~/test_spaxel_api.mpf'
        self._files_created.append(spaxel_path)

        path_saved = spaxel.save(spaxel_path, overwrite=True)
        assert os.path.exists(path_saved)
        assert os.path.realpath(os.path.expanduser(spaxel_path)), path_saved

        del spaxel

        spaxel_restored = marvin.tools.spaxel.Spaxel.restore(spaxel_path)
        assert spaxel_restored is not None
        assert isinstance(spaxel_restored, marvin.tools.spaxel.Spaxel)

        assert spaxel_restored.cube is not None
        assert isinstance(spaxel_restored.cube, Cube)
        assert spaxel_restored.cube.data_origin == 'api'
        assert spaxel_restored.cube.data is None
        assert spaxel_restored.cube.header['VERSDRP3'] == 'v2_0_1'

        assert spaxel_restored.maps is not None
        assert isinstance(spaxel_restored.maps, Maps)
        assert spaxel_restored.maps.data_origin == 'api'
        assert spaxel_restored.maps.data is None

        assert spaxel_restored.modelcube is not None
        assert isinstance(spaxel_restored.modelcube, marvin.tools.modelcube.ModelCube)
        assert spaxel_restored.modelcube.data_origin == 'api'
        assert spaxel_restored.modelcube.data is None


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
