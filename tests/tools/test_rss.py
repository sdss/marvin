#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2018-07-24
# @Filename: test_rss.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-08-04 13:35:39

import astropy.io.fits
import astropy.table
import numpy
import pytest

import marvin

from ..conftest import Galaxy, set_the_config


@pytest.fixture(scope='session')
def galaxy(get_params, plateifu):
    """Yield an instance of a Galaxy object for use in tests."""

    release, bintype, template = get_params

    set_the_config(release)

    gal = Galaxy(plateifu=plateifu)
    gal.set_params(bintype=bintype, template=template, release=release)
    gal.set_filepaths()
    gal.set_galaxy_data()

    yield gal


@pytest.fixture(scope='session')
def rss_session(galaxy, mode):
    # These get created only once per session.

    # if mode == 'auto' or str(galaxy.bintype) != 'SPX':
    #     pytest.skip()

    if mode == 'local':
        rss = marvin.tools.RSS(filename=galaxy.rsspath, release=galaxy.release, mode='local')
    else:
        rss = marvin.tools.RSS(plateifu=galaxy.plateifu, release=galaxy.release, mode='remote')
    rss.expdata = galaxy.rss
    yield rss


@pytest.fixture(scope='function')
def rss(rss_session):

    # In some of the tests we modify the RSS objects. Here we implement
    # a setup procedure that "unloads" the RSSFiber objects and resets the
    # autoload attribute.

    for rssfiber in rss_session:
        rssfiber.loaded = False

    rss_session.autoload = True

    yield rss_session


@pytest.fixture(scope='session')
def rssfiber(rss_session):

    fiberid = 0

    if rss_session[fiberid].loaded is False:
        rss_session[fiberid].load()

    yield rss_session[fiberid]


@pytest.mark.usefixtures('monkeyauth')
class TestRSS(object):

    def test_rss_init(self, rss):

        assert isinstance(rss, marvin.tools.RSS)
        assert isinstance(rss, marvin.tools.mixins.NSAMixIn)
        assert isinstance(rss, list)

        assert isinstance(rss.obsinfo, astropy.table.Table)

        if rss.mode == 'file':
            assert isinstance(rss.data, astropy.io.fits.HDUList)

        assert rss._wavelength is not None
        assert len(rss) == rss._nfibers

        rss.autoload = False  # To make things faster for this test
        assert all([isinstance(rss_fiber, marvin.tools.rss.RSSFiber) for rss_fiber in rss])

    @pytest.mark.parametrize('autoload', [True, False])
    def test_rss_autoload(self, rss, autoload):

        rss.autoload = autoload
        assert rss[0].loaded is autoload

    def test_load(self, rss):

        rss.autoload = False
        assert rss[0].loaded is False

        rss[0].load()
        assert rss[0].loaded is True

    def test_load_all(self, rss):

        if rss.mode == 'remote':
            pytest.skip()

        rss.load_all()
        assert all([rss_fiber.loaded is True for rss_fiber in rss])

    def test_obsinfo_to_rssfiber(self, rss):

        # We get it in this complicated way so that it is a different way of
        # obtianing it than in the _populate_fibres method.
        ifusize = int(str(rss.ifu)[0:-2])

        exp_idx = 0
        n_fiber = 1
        for rssfiber in rss:

            assert numpy.all(rss.obsinfo[exp_idx] == rssfiber.obsinfo)

            n_fiber += 1
            if n_fiber > ifusize:
                n_fiber = 1
                exp_idx += 1

    def test_getcube(self, rss):

        cube = rss.getCube()

        assert isinstance(cube, marvin.tools.Cube)
        assert cube.mode == rss.mode
        assert cube.plateifu == rss.plateifu
        assert cube.mangaid == rss.mangaid
        assert cube.release == rss.release

    def test_select_fibers(self, rss):

        # Skipping for API or it will take forever. Should not matter since
        # we have already tested slicing for API.
        if rss.data_origin == 'api':
            pytest.skip()

        fibers_expnum = rss.select_fibers(exposure_no=rss.expdata['expnum'])
        assert len(fibers_expnum) == rss.expdata['nfiber']
        assert fibers_expnum[0].obsinfo['EXPNUM'][0] == rss.expdata['expnum']

        fibers_mjd = rss.select_fibers(mjd=1234)
        assert len(fibers_mjd) == 0

        fibers_mjd = rss.select_fibers(mjd=rss.expdata['mjd'])
        assert len(fibers_mjd) == (rss.expdata['nexp'] * rss.expdata['nfiber'])
        assert fibers_mjd[0].obsinfo['MJD'][0] == rss.expdata['mjd']



@pytest.mark.usefixtures('monkeyauth')
class TestRSSFiber(object):

    def test_rssfiber_spectra(self, rssfiber):

        assert isinstance(rssfiber, marvin.tools.RSSFiber)
        assert isinstance(rssfiber.rss, marvin.tools.RSS)

        assert isinstance(rssfiber.obsinfo, astropy.table.Table)

        assert hasattr(rssfiber, 'ivar')
        assert isinstance(rssfiber.ivar, numpy.ndarray)
        assert len(rssfiber.ivar) == len(rssfiber.wavelength)

        assert hasattr(rssfiber, 'mask')
        assert isinstance(rssfiber.mask, numpy.ndarray)
        assert len(rssfiber.mask) == len(rssfiber.wavelength)

        for dm_element in rssfiber.rss.datamodel.rss + rssfiber.rss.datamodel.spectra:

            if dm_element.name == 'flux':
                continue

            spectrum = getattr(rssfiber, dm_element.name, None)
            assert spectrum is not None
            assert isinstance(spectrum, numpy.ndarray)
            assert len(spectrum) == len(rssfiber.wavelength)

    def test_rssfiber_data(self, rssfiber):

        rss_filename = rssfiber.rss._getFullPath()
        rss_hdu = astropy.io.fits.open(rss_filename)

        numpy.testing.assert_allclose(rss_hdu['FLUX'].data[rssfiber.fiberid, :], rssfiber.value)
        numpy.testing.assert_allclose(rss_hdu['IVAR'].data[rssfiber.fiberid, :], rssfiber.ivar)
        numpy.testing.assert_array_equal(rss_hdu['MASK'].data[rssfiber.fiberid, :], rssfiber.mask)

        for dm_element in rssfiber.rss.datamodel.rss:
            if dm_element.name == 'flux':
                continue
            fits_data = rss_hdu[dm_element.fits_extension()].data[rssfiber.fiberid, :]
            numpy.testing.assert_allclose(fits_data, getattr(rssfiber, dm_element.name).value)

        for dm_element in rssfiber.rss.datamodel.spectra:
            fits_data = rss_hdu[dm_element.fits_extension()].data
            numpy.testing.assert_allclose(fits_data, getattr(rssfiber, dm_element.name).value)

    def test_rssfiber_slice(self, rssfiber):

        n_elements = 10

        sliced = rssfiber[0:n_elements]

        assert len(sliced.value) == n_elements
        numpy.testing.assert_allclose(sliced.value, rssfiber.value[0:n_elements])

        assert len(sliced.ivar) == n_elements
        assert len(sliced.mask) == n_elements

        for dm_element in rssfiber.rss.datamodel.rss + rssfiber.rss.datamodel.spectra:

            if dm_element.name == 'flux':
                continue

            spectrum_sliced = getattr(sliced, dm_element.name, None)
            assert len(spectrum_sliced) == n_elements

        assert sliced.obsinfo is not None

    def test_rssfiber_masked(self, rssfiber):

        assert numpy.sum(rssfiber.masked.mask) > 0

    def test_rssfiber_descale(self, rssfiber):

        descaled = rssfiber.descale()
        numpy.testing.assert_allclose(descaled.value, rssfiber.value * rssfiber.unit.scale)

        assert descaled.obsinfo is not None


class TestPickling(object):

    def test_pickling_file(self, temp_scratch, rss):

        if rss.data_origin == 'file':
            assert rss.data is not None

        rss_file = temp_scratch / 'test_rss.mpf'
        rss.save(str(rss_file))
        assert rss_file.exists() is True

        rss_restored = marvin.tools.RSS.restore(str(rss_file))

        assert rss_restored.data_origin == rss.data_origin
        assert isinstance(rss_restored, marvin.tools.RSS)

        assert len(rss_restored) > 0
        assert isinstance(rss_restored[0], marvin.tools.RSSFiber)
        assert numpy.sum(rss_restored[0].value) > 0

        if rss.data_origin == 'file':
            assert rss_restored.data is not None
        else:
            assert rss_restored.data is None
