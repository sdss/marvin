#!/usr/bin/env python3
# encoding: utf-8
#
# datamodel.py
#
# Created by José Sánchez-Gallego on 18 Sep 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from marvin import config


__all__ = ('MapsProperty', 'MapsPropertyList', 'dap_datamodel',
           'get_dap_datamodel', 'get_dap_maplist', 'get_default_mapset')


class MapsPropertyList(list):

    def __init__(self, items, version=None):

        list.__init__(self, items)
        self.version = version

    def get(self, value):
        """Returns the MapsProperty and channel matching a value."""

        for item in self:
            channels = [None] if not item.channels else item.channels
            for channel in channels:
                if value.lower() == item.fullname(channel):
                    return item, channel

        return None

    def list_names(self):
        """Returns a list of property names."""

        return [prop.name for prop in self]

    def __eq__(self, name):
        for item in self:
            if name.lower() == item.name:
                return item

    def __contains__(self, name):
        for item in self:
            if name.lower() == item.name:
                return True
        return False

    def __getitem__(self, name):
        if isinstance(name, str):
            return self == name
        else:
            return list.__getitem__(self, name)


class MapsProperty(object):
    """A class to represent a Maps property."""

    def __init__(self, name, ivar=False, mask=False, channels=None, unit=None, description=''):

        self.name = name.lower()
        self.ivar = ivar
        self.mask = mask
        self.channels = channels
        self.unit = unit
        self.description = description

        if self.channels:
            for ii in range(len(self.channels)):
                self.channels[ii] = self.channels[ii].lower()

    def __repr__(self):

        return ('<MapsProperty name={0.name}, ivar={0.ivar}, mask={0.mask}, n_channels={1}>'
                .format(self, len(self.channels) if self.channels else None))

    def fullname(self, channel=None, ext=None):

        if self.channels is None:
            if ext:
                return self.name + '_' + ext
            else:
                return self.name

        if channel is None:
            raise ValueError('this MapsProperty has multiple channels. Please, specify one.')

        if channel.lower() not in self.channels:
            raise ValueError('invalid channel.')

        if ext is not None:
            assert ext in ['ivar', 'mask'], 'ext must be one of ivar or mask.'
            return '{0}_{1}_{2}'.format(self.name, ext, channel.lower())

        return '{0}_{1}'.format(self.name, channel.lower())


MPL4_emline_channels = ['oiid_3728', 'hb_4862', 'oiii_4960', 'oiii_5008', 'oi_6302',
                        'oi_6365', 'nii_6549', 'ha_6564', 'nii_6585', 'sii_6718', 'sii_6732']

MPL4_specindex_channels = ['d4000', 'caii0p39', 'hdeltaa', 'cn1', 'cn2', 'ca4227', 'hgammaa',
                           'fe4668', 'hb', 'mgb', 'fe5270', 'fe5335', 'fe5406', 'nad', 'tio1',
                           'tio2', 'nai0p82', 'caii0p86a', 'caii0p86b', 'caii0p86c', 'mgi0p88',
                           'tio0p89', 'feh0p99']

MPL4_specindex_units = ['Angstrom', 'Angstrom', 'Angstrom', 'mag', 'mag', 'Angstrom', 'Angstrom',
                        'Angstrom', 'Angstrom', 'Angstrom', 'Angstrom', 'Angstrom', 'Angstrom',
                        'Angstrom', 'mag', 'mag', 'Angstrom', 'Angstrom', 'Angstrom', 'Angstrom',
                        'Angstrom', 'Angstrom', 'Angstrom']

MPL5_extra_channels = ['oii_3727', 'oii_3729', 'heps_3971', 'hdel_4102', 'hgam_4341', 'heii_4687',
                       'hei_5877', 'siii_8831', 'siii_9071', 'siii_9533']

default_version = '2.0.2'

dap_datamodel = {

    '1.1.1': MapsPropertyList([
        MapsProperty('emline_gflux', ivar=True, mask=True, channels=MPL4_emline_channels,
                     unit='1E-17 erg/s/cm^2/spaxel',
                     description='Fluxes of emission lines based on a single Gaussian fit.'),
        MapsProperty('emline_gvel', ivar=True, mask=True, channels=MPL4_emline_channels,
                     unit='km/s',
                     description='Doppler velocity shifts for emission lines relative to '
                                 'the NSA redshift based on a single Gaussian fit.'),
        MapsProperty('emline_gsigma', ivar=True, mask=True, channels=MPL4_emline_channels,
                     unit='km/s',
                     description='Velocity dispersions of emission lines based on a '
                                 'single Gaussian fit.'),
        MapsProperty('emline_instsigma', ivar=False, mask=False,
                     channels=MPL4_emline_channels,
                     unit='km/s',
                     description='Instrumental velocity dispersion at the line centroids '
                                 'for emission lines (based on a single Gaussian fit.'),
        MapsProperty('emline_ew', ivar=True, mask=True, channels=MPL4_emline_channels,
                     unit='Angstrom',
                     description='Equivalent widths for emission lines based on a '
                                 'single Gaussian fit.'),
        MapsProperty('emline_sflux', ivar=True, mask=True, channels=MPL4_emline_channels,
                     unit='1E-17 erg/s/cm^2/spaxel',
                     description='Fluxes for emission lines based on integrating the '
                                 'flux over a set of passbands.'),
        MapsProperty('stellar_vel', ivar=True, mask=True, channels=None,
                     unit='km/s',
                     description='Stellar velocity measurements.'),
        MapsProperty('stellar_sigma', ivar=True, mask=True, channels=None,
                     unit='km/s',
                     description='Stellar velocity dispersion measurements.'),
        MapsProperty('specindex', ivar=True, mask=True,
                     channels=MPL4_specindex_channels,
                     unit=None,
                     description='Measurements of spectral indices.'),
        MapsProperty('binid', ivar=False, mask=False, channels=None,
                     unit=None,
                     description='ID number for the bin for which the pixel value was '
                                 'calculated; bins are sorted by S/N.')],
        version='1.1.1'),

    '2.0.2': MapsPropertyList([
        MapsProperty('spx_skycoo', ivar=False, mask=False, channels=['on_sky_x', 'on_sky_y'],
                     unit='arcsec',
                     description='Offsets of each spaxel from the galaxy center.'),
        MapsProperty('spx_ellcoo', ivar=False, mask=False,
                     channels=['elliptical_radius', 'elliptical_azimuth'],
                     unit=['arcsec', 'degrees'],
                     description='Elliptical polar coordinates of each spaxel from '
                                 'the galaxy center.'),
        MapsProperty('spx_mflux', ivar=True, mask=False, channels=None,
                     unit='1E-17 erg/s/cm^2/ang/spaxel',
                     description='Mean flux in r-band (5600.1-6750.0 ang).'),
        MapsProperty('spx_snr', ivar=False, mask=False, channels=None,
                     unit=None,
                     description='r-band signal-to-noise ratio per pixel.'),
        MapsProperty('binid', ivar=False, mask=False, channels=None,
                     unit=None,
                     description='Numerical ID for spatial bins.'),
        MapsProperty('bin_lwskycoo', ivar=False, mask=False,
                     channels=['lum_weighted_on_sky_x', 'lum_weighted_on_sky_y'],
                     unit='arcsec',
                     description='Light-weighted offset of each bin from the galaxy center.'),
        MapsProperty('bin_lwellcoo', ivar=False, mask=False,
                     channels=['lum_weighted_elliptical_radius',
                               'lum_weighted_elliptical_azimuth'],
                     unit=['arcsec', 'degrees'],
                     description='light-weighted elliptical polar coordinates of each bin '
                                 'from the galaxy center.'),
        MapsProperty('bin_area', ivar=False, mask=False, channels=None,
                     unit='arcsec^2',
                     description='Area of each bin.'),
        MapsProperty('bin_farea', ivar=False, mask=False, channels=None,
                     unit=None,
                     description='Fractional area that the bin covers for the expected bin '
                                 'shape (only relevant for radial binning).'),
        MapsProperty('bin_mflux', ivar=True, mask=True, channels=None,
                     unit='1E-17 erg/s/cm^2/ang/spaxel',
                     description='Mean flux in the r-band for the binned spectra.'),
        MapsProperty('bin_snr', ivar=False, mask=False, channels=None,
                     unit=None,
                     description='r-band signal-to-noise ratio per pixel in the binned spectra.'),
        MapsProperty('stellar_vel', ivar=True, mask=True, channels=None,
                     unit='km/s',
                     description='Stellar velocity relative to NSA redshift.'),
        MapsProperty('stellar_sigma', ivar=True, mask=True, channels=None,
                     unit='km/s',
                     description='Stellar velocity dispersion (must be corrected using '
                                 'STELLAR_SIGMACORR)'),
        MapsProperty('stellar_sigmacorr', ivar=False, mask=False, channels=None,
                     unit='km/s',
                     description='Quadrature correction for STELLAR_SIGMA to obtain the '
                                 'astrophysical velocity dispersion.)'),
        MapsProperty('stellar_cont_fresid', ivar=False, mask=False,
                     channels=['68th_percentile', '99th_percentile'],
                     unit=None,
                     description='68%% and 99%% growth of the fractional residuals between '
                                 'the model and data'),
        MapsProperty('stellar_cont_rchi2', ivar=False, mask=False, channels=None,
                     unit=None,
                     description='Reduced chi-square of the stellar continuum fit.'),
        MapsProperty('emline_sflux', ivar=True, mask=True,
                     channels=MPL4_emline_channels + MPL5_extra_channels,
                     unit='1E-17 erg/s/cm^2/spaxel',
                     description='Non-parametric summed flux for emission lines.'),
        MapsProperty('emline_sew', ivar=True, mask=True,
                     channels=MPL4_emline_channels + MPL5_extra_channels,
                     unit='Angstrom',
                     description='Emission line non-parametric equivalent widths measurements.'),
        MapsProperty('emline_gflux', ivar=True, mask=True,
                     channels=MPL4_emline_channels + MPL5_extra_channels,
                     unit='1E-17 erg/s/cm^2/spaxel',
                     description='Gaussian profile integrated flux for emission lines.'),
        MapsProperty('emline_gvel', ivar=True, mask=True,
                     channels=MPL4_emline_channels + MPL5_extra_channels,
                     unit='km/s',
                     description='Gaussian profile velocity for emission lines.'),
        MapsProperty('emline_gsigma', ivar=True, mask=True,
                     channels=MPL4_emline_channels + MPL5_extra_channels,
                     unit='km/s',
                     description='Gaussian profile velocity dispersion for emission lines; '
                                 'must be corrected using EMLINE_INSTSIGMA'),
        MapsProperty('emline_instsigma', ivar=False, mask=False,
                     channels=MPL4_emline_channels + MPL5_extra_channels,
                     unit='km/s',
                     description='Instrumental dispersion at the fitted line center.'),
        MapsProperty('specindex', ivar=True, mask=True,
                     channels=['d4000', 'dn4000'],
                     unit=None,
                     description='Measurements of spectral indices.'),
        MapsProperty('specindex_corr', ivar=False, mask=False,
                     channels=['d4000', 'dn4000'],
                     unit=None,
                     description='Velocity dispersion corrections for the '
                                 'spectral index measurements '
                                 '(can be ignored for D4000, Dn4000).')],
        version='2.0.2')}


def get_dap_datamodel(dapver=None):
    """Returns the correct DAP datamodel for dapver."""

    if not dapver:
        __, dapver = config.lookUpVersions(config.release)

    if dapver not in dap_datamodel:
        return dap_datamodel[default_version]
    else:
        return dap_datamodel[dapver]


def get_dap_maplist(dapver=None, web=None):
    ''' Returns a list of all possible maps for dapver '''

    dapdm = get_dap_datamodel(dapver)
    daplist = []

    for p in dapdm:
        if p.channels:
            if web:
                daplist.extend(['{0}:{1}'.format(p.name, c) for c in p.channels])
            else:
                daplist.extend(['{0}_{1}'.format(p.name, c) for c in p.channels])
        else:
            daplist.append(p.name)

    return daplist


def get_default_mapset(dapver=None):
    ''' Returns a list of six default maps for display '''

    dapdefaults = {
        # 6 defaults
        # '1.1.1': ['emline_gflux:oiid_3728', 'emline_gflux:hb_4862', 'emline_gflux:oiii_5008',
        #           'emline_gflux:ha_6564', 'emline_gflux:nii_6585', 'emline_gflux:sii_6718'],
        # '2.0.2': ['emline_gflux:oiid_3728', 'emline_gflux:hb_4862', 'emline_gflux:oiii_5008',
        #           'emline_gflux:ha_6564', 'emline_gflux:nii_6585', 'emline_gflux:sii_6718']
        # 3 defaults
        '1.1.1': ['stellar_vel', 'emline_gflux:ha_6564', 'specindex:d4000'],
        '2.0.2': ['stellar_vel', 'emline_gflux:ha_6564', 'specindex:d4000']
    }

    return dapdefaults[dapver] if dapver in dapdefaults else []


