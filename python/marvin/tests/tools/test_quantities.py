#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2018-07-20
# @Filename: test_quantities.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by:   andrews
# @Last modified time: 2018-10-19 14:10:15


import matplotlib
import numpy
import pytest
from astropy import units as u

from marvin.tests import marvin_test_if
from marvin.tools.quantities import DataCube, Spectrum


spaxel_unit = u.Unit('spaxel', represents=u.pixel, doc='A spectral pixel', parse_strict='silent')


@pytest.fixture(scope='function')
def datacube():
    """Produces a simple 3D array for datacube testing."""

    flux = numpy.tile([numpy.arange(1, 1001, dtype=numpy.float32)],
                      (100, 1)).T.reshape(1000, 10, 10)
    ivar = (1. / (flux / 100))**2
    mask = numpy.zeros(flux.shape, dtype=numpy.int)
    wave = numpy.arange(1, 1001)

    redcorr = numpy.ones(1000) * 1.5

    mask[50:100, 5, 5] = 2**10
    mask[500:600, 3, 3] = 2**4

    scale = 1e-3

    datacube = DataCube(flux, wave, ivar=ivar, mask=mask, redcorr=redcorr, scale=scale,
                        unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit,
                        pixmask_flag='MANGA_DRP3PIXMASK')

    yield datacube


@pytest.fixture(scope='function')
def spectrum():
    """Produces a simple 1D array for datacube testing."""

    flux = numpy.arange(1, 1001, dtype=numpy.float32)
    ivar = (1. / (flux / 100))**2
    mask = numpy.zeros(flux.shape, dtype=numpy.int)
    wave = numpy.arange(1, 1001)

    mask[50:100] = 2**10
    mask[500:600] = 2**4

    scale = 1e-3

    datacube = Spectrum(flux, wave, ivar=ivar, mask=mask, scale=scale,
                        unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit,
                        pixmask_flag='MANGA_DRP3PIXMASK')

    yield datacube


class TestDataCube(object):

    def test_datacube(self, datacube):

        assert datacube.value is not None
        assert datacube.ivar is not None
        assert datacube.mask is not None

        numpy.testing.assert_array_equal(datacube.value.shape, datacube.ivar.shape)
        numpy.testing.assert_array_equal(datacube.value.shape, datacube.mask.shape)

        assert datacube.pixmask is not None

    def test_masked(self, datacube):

        assert isinstance(datacube.masked, numpy.ma.MaskedArray)
        assert numpy.sum(datacube.masked.mask) == 50

        datacube.pixmask_flag = None
        assert numpy.sum(datacube.masked.mask) == 150

    def test_snr(self, datacube):

        assert datacube.snr[100, 5, 5] == pytest.approx(100)

    def test_error(self, datacube):

        numpy.testing.assert_almost_equal(datacube.error.value, numpy.sqrt(1 / datacube.ivar))
        assert datacube.error.unit == datacube.unit

        numpy.testing.assert_almost_equal(datacube.error.value, datacube.std.value)

    def test_descale(self, datacube):

        assert datacube.unit.scale == 1e-3

        descaled = datacube.descale()
        datacube.unit.scale == 1

        numpy.testing.assert_almost_equal(descaled.value, datacube.value * datacube.unit.scale)
        numpy.testing.assert_almost_equal(descaled.ivar, datacube.ivar / datacube.unit.scale**2)

    def test_redcorr(self, datacube):

        der = datacube.deredden()
        assert isinstance(der, DataCube)

        numpy.testing.assert_allclose(der.value, datacube.value * 1.5)
        numpy.testing.assert_allclose(der.ivar, datacube.ivar / 1.5**2)
        numpy.testing.assert_allclose(der.mask, datacube.mask)

        assert der.redcorr is None
        assert der.pixmask_flag == datacube.pixmask_flag

        new_redcorr = (numpy.ones(1000) * 2.)
        new_der = datacube.deredden(redcorr=new_redcorr)

        numpy.testing.assert_allclose(new_der.value, datacube.value * 2)
        numpy.testing.assert_allclose(new_der.ivar, datacube.ivar / 2**2)

        datacube.redcorr = None
        with pytest.raises(ValueError):
            datacube.deredden()

    def test_slice_datacube(self, datacube):

        new_datacube = datacube[:, 3:5, 3:5]

        assert isinstance(new_datacube, DataCube)
        numpy.testing.assert_almost_equal(new_datacube.value, datacube.value[:, 3:5, 3:5])
        numpy.testing.assert_almost_equal(new_datacube.ivar, datacube.ivar[:, 3:5, 3:5])
        numpy.testing.assert_almost_equal(new_datacube.mask, datacube.mask[:, 3:5, 3:5])
        numpy.testing.assert_almost_equal(new_datacube.redcorr, datacube.redcorr)
        assert new_datacube.pixmask_flag == datacube.pixmask_flag

    def test_slice_wave(self, datacube):

        new_datacube = datacube[10:100]

        assert isinstance(new_datacube, DataCube)
        numpy.testing.assert_almost_equal(new_datacube.value, datacube.value[10:100, :, :])
        numpy.testing.assert_almost_equal(new_datacube.ivar, datacube.ivar[10:100, :, :])
        numpy.testing.assert_almost_equal(new_datacube.mask, datacube.mask[10:100, :, :])
        numpy.testing.assert_almost_equal(new_datacube.redcorr, datacube.redcorr[10:100])
        assert new_datacube.pixmask_flag == datacube.pixmask_flag

    def test_slice_spectrum(self, datacube):

        new_spectrum = datacube[:, 5, 5]

        assert isinstance(new_spectrum, Spectrum)
        numpy.testing.assert_almost_equal(new_spectrum.value, datacube.value[:, 5, 5])
        numpy.testing.assert_almost_equal(new_spectrum.ivar, datacube.ivar[:, 5, 5])
        numpy.testing.assert_almost_equal(new_spectrum.mask, datacube.mask[:, 5, 5])
        assert new_spectrum.pixmask_flag == datacube.pixmask_flag

    @marvin_test_if(mark='include', cube={'plateifu': '8485-1901',
                                          'data_origin': 'file',
                                          'initial_mode': 'local'})
    def test_cube_quantities(self, cube):

        assert cube.flux is not None

        assert isinstance(cube.flux, numpy.ndarray)
        assert isinstance(cube.flux, DataCube)

        assert isinstance(cube.spectral_resolution, Spectrum)

        if cube.release in ['MPL-4', 'MPL-5']:
            with pytest.raises(AssertionError) as ee:
                cube.spectral_resolution_prepixel
            assert 'spectral_resolution_prepixel is not present in his MPL version' in str(ee)
        else:
            assert isinstance(cube.spectral_resolution_prepixel, Spectrum)

        assert cube.flux.pixmask.values_to_bits(3) == [0, 1]

        assert cube.flux.pixmask.values_to_labels(3) == ['NOCOV', 'LOWCOV']

    @pytest.mark.parametrize('names, expected', [(['NOCOV', 'LOWCOV'], 3),
                                                 ('DONOTUSE', 1024)])
    def test_labels_to_value(self, cube, names, expected):
        assert cube.flux.pixmask.labels_to_value(names) == expected

    @marvin_test_if(mark='include', modelcube={'plateifu': '8485-1901',
                                               'data_origin': 'file',
                                               'initial_mode': 'local'})
    def test_modelcube_quantities(self, modelcube):

        for mc in modelcube.datamodel:
            if hasattr(modelcube, mc.name):
                modelcube_quantity = getattr(modelcube, mc.name)
                assert isinstance(modelcube_quantity, DataCube)
                assert modelcube_quantity.pixmask_flag == 'MANGA_DAPSPECMASK'


class TestSpectrum(object):

    def test_spectrum(self, spectrum):

        assert spectrum.value is not None
        assert spectrum.ivar is not None
        assert spectrum.mask is not None

        numpy.testing.assert_array_equal(spectrum.value.shape, spectrum.ivar.shape)
        numpy.testing.assert_array_equal(spectrum.value.shape, spectrum.mask.shape)

        assert spectrum.pixmask is not None

    def test_masked(self, spectrum):

        assert isinstance(spectrum.masked, numpy.ma.MaskedArray)
        assert numpy.sum(spectrum.masked.mask) == 50

        spectrum.pixmask_flag = None
        assert numpy.sum(spectrum.masked.mask) == 150

    def test_snr(self, spectrum):

        assert spectrum.snr[100] == pytest.approx(100)

    def test_error(self, spectrum):

        numpy.testing.assert_almost_equal(spectrum.error.value, numpy.sqrt(1 / spectrum.ivar))
        assert spectrum.error.unit == spectrum.unit

        numpy.testing.assert_almost_equal(spectrum.error.value, spectrum.std.value)

    def test_descale(self, spectrum):

        assert spectrum.unit.scale == 1e-3

        descaled = spectrum.descale()
        spectrum.unit.scale == 1

        numpy.testing.assert_almost_equal(descaled.value, spectrum.value * spectrum.unit.scale)
        numpy.testing.assert_almost_equal(descaled.ivar, spectrum.ivar / spectrum.unit.scale**2)

    def test_slice_spectrum(self, spectrum):

        new_spectrum = spectrum[10:100]

        assert isinstance(new_spectrum, Spectrum)
        numpy.testing.assert_almost_equal(new_spectrum.value, spectrum.value[10:100])
        numpy.testing.assert_almost_equal(new_spectrum.ivar, spectrum.ivar[10:100])
        numpy.testing.assert_almost_equal(new_spectrum.mask, spectrum.mask[10:100])
        assert new_spectrum.pixmask_flag == spectrum.pixmask_flag

    @marvin_test_if(mark='include', cube={'plateifu': '8485-1901',
                                          'data_origin': 'file',
                                          'initial_mode': 'local'})
    def test_cube_quantities(self, cube):

        for sp in cube.datamodel.spectra:
            cube_quantity = getattr(cube, sp.name)
            assert isinstance(cube_quantity, Spectrum)
            assert cube_quantity.pixmask_flag is None

    def test_plot(self, spectrum):

        ax = spectrum.plot(show_std=True)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_no_std_no_mask(self):
        sp = Spectrum(numpy.random.randn(1000), wavelength=numpy.arange(1000))
        sp.plot()

    def test_plot_no_std(self):

        mask = numpy.zeros(1000, dtype=numpy.int)
        mask[50:100] = 2**10
        mask[500:600] = 2**4

        sp = Spectrum(
            flux=numpy.random.randn(1000),
            wavelength=numpy.arange(1000),
            mask=mask,
            pixmask_flag='MANGA_DRP3PIXMASK',
        )
        sp.plot()

    def test_plot_no_mask(self):
        flux = numpy.random.randn(1000)
        ivar = (1. / (flux / 100))**2

        sp = Spectrum(
            flux=flux,
            wavelength=numpy.arange(1000),
            ivar=ivar,
        )
        sp.plot()
