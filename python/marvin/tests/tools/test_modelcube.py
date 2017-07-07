#!/usr/bin/env python
# encoding: utf-8
#
# test_modelcube.py
#
# Created by José Sánchez-Gallego on 25 Sep 2016.


from __future__ import division, print_function, absolute_import

import os

import pytest
from astropy.io import fits
from astropy.wcs import WCS

from marvin import config
from marvin.core.exceptions import MarvinError
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube
from marvin.tests import marvin_test_if


class TestModelCubeInit(object):

    def _test_init(self, model_cube, galaxy, bintype='SPX', template_kin='GAU-MILESHC'):
        assert model_cube._release == galaxy.release
        assert model_cube._drpver == galaxy.drpver
        assert model_cube._dapver == galaxy.dapver
        assert model_cube.bintype == bintype
        assert model_cube.template_kin == template_kin
        assert model_cube.plateifu == galaxy.plateifu
        assert model_cube.mangaid == galaxy.mangaid
        assert isinstance(model_cube.header, fits.Header)
        assert isinstance(model_cube.wcs, WCS)
        assert model_cube.wavelength is not None
        assert model_cube.redcorr is not None

    @pytest.mark.parametrize('data_origin', ['file', 'db', 'api'])
    def test_init_modelcube(self, galaxy, data_origin):
        if data_origin == 'file':
            kwargs = {'filename': galaxy.modelpath}
        elif data_origin == 'db':
            kwargs = {'plateifu': galaxy.plateifu}
        elif data_origin == 'api':
            kwargs = {'plateifu': galaxy.plateifu, 'mode': 'remote'}

        model_cube = ModelCube(**kwargs)
        assert model_cube.data_origin == data_origin
        self._test_init(model_cube, galaxy)

    # TODO remove set_tmp_mpl
    # def test_init_from_file_global_mpl4(self, galaxy):
    #     with set_tmp_mpl('MPL-4'):
    #         config.setMPL('MPL-4')
    #         model_cube = ModelCube(filename=galaxy.modelpath)
    #         assert model_cube.data_origin == 'file'
    #         self._test_init(model_cube, galaxy)

    # TODO remove set_tmp_mpl
    # def test_raises_exception_mpl4(self, galaxy):
    #     with set_tmp_mpl('MPL-4'):
    #         config.setMPL('MPL-4')
    #         with pytest.raises(MarvinError) as cm:
    #             ModelCube(plateifu=galaxy.plateifu)
    #         assert 'ModelCube requires at least dapver=\'2.0.2\'' in str(cm.value)

    @pytest.mark.parametrize('data_origin', ['file', 'db', 'api'])
    def test_init_modelcube_bintype(self, galaxy, data_origin):
        kwargs = {'bintype': galaxy.bintype}
        if data_origin == 'file':
            kwargs['filename'] = galaxy.modelpath
        elif data_origin == 'db':
            kwargs['plateifu'] = galaxy.plateifu
        elif data_origin == 'api':
            kwargs['plateifu'] = galaxy.plateifu
            kwargs['mode'] = 'remote'

        model_cube = ModelCube(**kwargs)
        assert model_cube.data_origin == data_origin
        self._test_init(model_cube, galaxy, bintype=galaxy.bintype)

class TestModelCube(object):

    def test_get_flux_db(self, galaxy):
        model_cube = ModelCube(plateifu=galaxy.plateifu)
        assert model_cube.flux.shape == (4563, 34, 34)

    def test_get_flux_api_raises_exception(self, galaxy):
        model_cube = ModelCube(plateifu=galaxy.plateifu, mode='remote')
        with pytest.raises(MarvinError) as cm:
            model_cube.flux
        assert 'cannot return a full cube in remote mode.' in str(cm.value)

    def test_get_cube_file(self, galaxy):
        model_cube = ModelCube(filename=galaxy.modelpath)
        assert isinstance(model_cube.cube, Cube)

    def test_get_maps_api(self, galaxy):
        model_cube = ModelCube(plateifu=galaxy.plateifu, mode='remote')
        assert isinstance(model_cube.maps, Maps)
    
    @marvin_test_if(mark='include', data_origin=['db', 'api'])
    @marvin_test_if(mark='skip', galaxy=dict(release=['MPL-4']))
    def test_modelcube_redshift(self, galaxy, mode):
        model_cube = ModelCube(plateifu=galaxy.plateifu, mode=mode)
        assert pytest.approx(model_cube.nsa.z, galaxy.redshift)
    
    @marvin_test_if(mark='include', data_origin=['file'])
    @marvin_test_if(mark='skip', galaxy=dict(release=['MPL-4']))
    def test_modelcube_redshift_file(self, galaxy):
        
        # TODO Remove
        files_to_download = ['manga-7443-12701-LOGCUBE-NRE-GAU-MILESHC.fits.gz',
                             'manga-7443-12701-LOGCUBE-ALL-GAU-MILESHC.fits.gz',
                             'manga-7443-12701-LOGCUBE-SPX-GAU-MILESHC.fits.gz',
                             'manga-7443-12701-LOGCUBE-VOR10-GAU-MILESHC.fits.gz']
        if galaxy.modelpath.split('/')[-1] in files_to_download:
            pytest.skip('Remove this skip once I download the files.')

        model_cube = ModelCube(filename=galaxy.modelpath)
        assert pytest.approx(model_cube.nsa.z, galaxy.redshift)




class TestGetSpaxel(object):

    def _test_getspaxel(self, spaxel, galaxy, bintype='SPX', template_kin='GAU-MILESHC'):
        assert spaxel._drpver == galaxy.drpver
        assert spaxel._dapver == galaxy.dapver
        assert spaxel.plateifu == galaxy.plateifu
        assert spaxel.mangaid == galaxy.mangaid
        assert spaxel.modelcube is not None
        assert spaxel.modelcube.bintype == bintype
        assert spaxel.modelcube.template_kin == template_kin
        assert spaxel._parent_shape == (34, 34)

        assert spaxel.model_flux is not None
        assert spaxel.model is not None
        assert spaxel.emline is not None
        assert spaxel.emline_base is not None
        assert spaxel.stellar_continuum is not None
        assert spaxel.redcorr is not None

    @pytest.mark.parametrize('data_origin', ['file', 'db','api'])
    def test_getspaxel(self, galaxy, data_origin):
        if data_origin == 'file':
            kwargs = {'filename': galaxy.modelpath}
        elif data_origin == 'db':
            kwargs = {'plateifu': galaxy.plateifu}
        elif data_origin == 'api':
            kwargs = {'plateifu': galaxy.plateifu, 'mode': 'remote'}

        model_cube = ModelCube(**kwargs)
        spaxel = model_cube.getSpaxel(x=1, y=2)
        self._test_getspaxel(spaxel, galaxy)

    def test_getspaxel_db_only_model(self, galaxy):

        model_cube = ModelCube(plateifu=galaxy.plateifu)
        spaxel = model_cube.getSpaxel(x=1, y=2, properties=False, spectrum=False)
        self._test_getspaxel(spaxel, galaxy)
        assert spaxel.cube is None
        assert spaxel.spectrum is None
        assert spaxel.maps is None
        assert len(spaxel.properties) == 0

    @marvin_test_if(mark='skip', galaxy=dict(release=['MPL-4']))
    @marvin_test_if(mark='skip', db=['nodb'])
    @pytest.mark.parametrize('mpl, flux, ivar, mask',
                             [('MPL-5', 0.016027471050620079, 361.13595581054693, 33)])
    def test_getspaxel_matches_file_db_remote(self, galaxy, mpl, flux, mask, ivar):

        # TODO move parametrized flux, ivar, and mask values to galaxy_test_data.dat
        
        # TODO Remove
        files_to_download = ['manga-7443-12701-LOGCUBE-NRE-GAU-MILESHC.fits.gz',
                             'manga-7443-12701-LOGCUBE-ALL-GAU-MILESHC.fits.gz',
                             'manga-7443-12701-LOGCUBE-SPX-GAU-MILESHC.fits.gz',
                             'manga-7443-12701-LOGCUBE-VOR10-GAU-MILESHC.fits.gz']
        if galaxy.modelpath.split('/')[-1] in files_to_download:
            pytest.skip('Remove this skip once I download the files.')

        modelcube_file = ModelCube(filename=galaxy.modelpath)
        modelcube_db = ModelCube(mangaid=galaxy.mangaid)
        modelcube_api = ModelCube(mangaid=galaxy.mangaid, mode='remote')

        config.setMPL(mpl)
        assert config.release == mpl

        assert modelcube_file.data_origin == 'file'
        assert modelcube_db.data_origin == 'db'
        assert modelcube_api.data_origin == 'api'

        xx = 12
        yy = 5
        idx = 200

        spaxel_slice_file = modelcube_file[yy, xx]
        spaxel_slice_db = modelcube_db[yy, xx]
        spaxel_slice_api = modelcube_api[yy, xx]

        assert pytest.approx(spaxel_slice_file.model_flux.flux[idx], flux)
        assert pytest.approx(spaxel_slice_db.model_flux.flux[idx], flux)
        assert pytest.approx(spaxel_slice_api.model_flux.flux[idx], flux)

        assert pytest.approx(spaxel_slice_file.model_flux.ivar[idx], ivar)
        assert pytest.approx(spaxel_slice_db.model_flux.ivar[idx], ivar)
        assert pytest.approx(spaxel_slice_api.model_flux.ivar[idx], ivar)

        assert pytest.approx(spaxel_slice_file.model_flux.mask[idx], mask)
        assert pytest.approx(spaxel_slice_db.model_flux.mask[idx], mask)
        assert pytest.approx(spaxel_slice_api.model_flux.mask[idx], mask)

        xx_cen = -5
        yy_cen = -12

        spaxel_getspaxel_file = modelcube_file.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_db = modelcube_db.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_api = modelcube_api.getSpaxel(x=xx_cen, y=yy_cen)

        assert pytest.approx(spaxel_getspaxel_file.model_flux.flux[idx], flux)
        assert pytest.approx(spaxel_getspaxel_db.model_flux.flux[idx], flux)
        assert pytest.approx(spaxel_getspaxel_api.model_flux.flux[idx], flux)

        assert pytest.approx(spaxel_getspaxel_file.model_flux.ivar[idx], ivar)
        assert pytest.approx(spaxel_getspaxel_db.model_flux.ivar[idx], ivar)
        assert pytest.approx(spaxel_getspaxel_api.model_flux.ivar[idx], ivar)

        assert pytest.approx(spaxel_getspaxel_file.model_flux.mask[idx], mask)
        assert pytest.approx(spaxel_getspaxel_db.model_flux.mask[idx], mask)
        assert pytest.approx(spaxel_getspaxel_api.model_flux.mask[idx], mask)


class TestPickling(object):

    def test_pickling_file(self, tmpfiles, galaxy):
        modelcube = ModelCube(filename=galaxy.modelpath, bintype=galaxy.bintype)
        assert modelcube.data_origin == 'file'
        assert isinstance(modelcube, ModelCube)
        assert modelcube.data is not None

        path = modelcube.save()
        tmpfiles.append(path)

        assert os.path.exists(path)
        assert os.path.realpath(path) == os.path.realpath(galaxy.modelpath[0:-7] + 'mpf')
        assert modelcube.data is not None

        modelcube = None
        assert modelcube is None

        modelcube_restored = ModelCube.restore(path)
        assert modelcube_restored.data_origin == 'file'
        assert isinstance(modelcube_restored, ModelCube)
        assert modelcube_restored.data is not None

    def test_pickling_file_custom_path(self, tmpfiles, galaxy):
        modelcube = ModelCube(filename=galaxy.modelpath, bintype=galaxy.bintype)
        assert modelcube.data_origin == 'file'
        assert isinstance(modelcube, ModelCube)
        assert modelcube.data is not None

        test_path = '~/test.mpf'
        assert not os.path.isfile(test_path)

        path = modelcube.save(path=test_path)
        tmpfiles.append(path)

        assert os.path.exists(path)
        assert path == os.path.realpath(os.path.expanduser(test_path))

        modelcube_restored = ModelCube.restore(path, delete=True)
        assert modelcube_restored.data_origin == 'file'
        assert isinstance(modelcube_restored, ModelCube)
        assert modelcube_restored.data is not None

        assert not os.path.exists(path)

    def test_pickling_db(self, galaxy):
        modelcube = ModelCube(plateifu=galaxy.plateifu, bintype=galaxy.bintype)

        with pytest.raises(MarvinError) as cm:
            modelcube.save()

        assert 'objects with data_origin=\'db\' cannot be saved.' in str(cm.value)

    def test_pickling_api(self, tmpfiles, galaxy):
        modelcube = ModelCube(plateifu=galaxy.plateifu, bintype=galaxy.bintype, mode='remote')
        assert modelcube.data_origin == 'api'
        assert isinstance(modelcube, ModelCube)
        assert modelcube.data is None

        path = modelcube.save()
        tmpfiles.append(path)

        assert os.path.exists(path)
        assert os.path.realpath(path) == os.path.realpath(galaxy.modelpath[0:-7] + 'mpf')

        modelcube = None
        assert modelcube is None

        modelcube_restored = ModelCube.restore(path)
        assert modelcube_restored.data_origin == 'api'
        assert isinstance(modelcube_restored, ModelCube)
        assert modelcube_restored.data is None
