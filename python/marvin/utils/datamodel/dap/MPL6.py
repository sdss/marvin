# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-13 16:05:56
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-08-06 11:45:33

from __future__ import absolute_import, division, print_function

from astropy import units as u

from marvin.utils.datamodel.maskbit import get_maskbits

from .base import Bintype, Channel, DAPDataModel, Model, MultiChannelProperty, Property
from .base import spaxel as spaxel_unit
from .MPL5 import ALL, GAU_MILESHC, NRE, SPX, VOR10


HYB10 = Bintype('HYB10', description='Binning and stellar continuum fitting as VOR10, '
                                     'but emission lines are fitted per spaxel.')

# The two lines in the OII doublet is fitted independently for gaussian
# measurements. In that case oii_3727 and oii_3729 are populated. For summed
# flux measurements, the lines cannot be separated so oiid_3728 contains
# the summed flux. In that case, oii_3729 is null and only kept to maintain`
# the number of channels constant.
oiid_channel = Channel('oiid_3728', formats={'string': 'OIId 3728',
                       'latex': r'$\forb{O\,IId}\;\lambda\lambda 3728$'}, idx=0)
oii_channel = Channel('oii_3727', formats={'string': 'OII 3727',
                      'latex': r'$\forb{O\,II}\;\lambda 3727$'}, idx=0)

MPL6_emline_channels = [
    Channel('oii_3729', formats={'string': 'OII 3729',
                                 'latex': r'$\forb{O\,II}\;\lambda 3729$'}, idx=1),
    Channel('hthe_3798', formats={'string': 'H-theta 3798',
                                  'latex': r'H$\theta\;\lambda 3798$'}, idx=2),
    Channel('heta_3836', formats={'string': 'H-eta 3836',
                                  'latex': r'H$\eta\;\lambda 3836$'}, idx=3),
    Channel('neiii_3869', formats={'string': 'NeIII 3869',
                                   'latex': r'$\forb{Ne\,III}\;\lambda 3869$'}, idx=4),
    Channel('hzet_3890', formats={'string': 'H-zeta 3890',
                                  'latex': r'H$\zeta\;\lambda 3890$'}, idx=5),
    Channel('neiii_3968', formats={'string': 'NeIII 3968',
                                   'latex': r'$\forb{Ne\,III}\;\lambda 3968$'}, idx=6),
    Channel('heps_3971', formats={'string': 'H-epsilon 3971',
                                  'latex': r'H$\epsilon\;\lambda 3971$'}, idx=7),
    Channel('hdel_4102', formats={'string': 'H-delta 4102',
                                  'latex': r'H$\delta\;\lambda 4102$'}, idx=8),
    Channel('hgam_4341', formats={'string': 'H-gamma 4341',
                                  'latex': r'H$\gamma\;\lambda 4341$'}, idx=9),
    Channel('heii_4687', formats={'string': 'HeII 4681',
                                  'latex': r'He\,II$\;\lambda 4687$'}, idx=10),
    Channel('hb_4862', formats={'string': 'H-beta 4862',
                                'latex': r'H$\beta\;\lambda 4862$'}, idx=11),
    Channel('oiii_4960', formats={'string': 'OIII 4960',
                                  'latex': r'$\forb{O\,III}\;\lambda 4960$'}, idx=12),
    Channel('oiii_5008', formats={'string': 'OIII 5008',
                                  'latex': r'$\forb{O\,III}\;\lambda 5008$'}, idx=13),
    Channel('hei_5877', formats={'string': 'HeI 5877',
                                 'latex': r'He\,I$\;\lambda 5877$'}, idx=14),
    Channel('oi_6302', formats={'string': 'OI 6302',
                                'latex': r'$\forb{O\,I}\;\lambda 6302$'}, idx=15),
    Channel('oi_6365', formats={'string': 'OI 6365',
                                'latex': r'$\forb{O\,I}\;\lambda 6365$'}, idx=16),
    Channel('nii_6549', formats={'string': 'NII 6549',
                                 'latex': r'$\forb{N\,II}\;\lambda 6549$'}, idx=17),
    Channel('ha_6564', formats={'string': 'H-alpha 6564',
                                'latex': r'H$\alpha\;\lambda 6564$'}, idx=18),
    Channel('nii_6585', formats={'string': 'NII 6585',
                                 'latex': r'$\forb{N\,II}\;\lambda 6585$'}, idx=19),
    Channel('sii_6718', formats={'string': 'SII 6718',
                                 'latex': r'$\forb{S\,II}\;\lambda 6718$'}, idx=20),
    Channel('sii_6732', formats={'string': 'SII 6732',
                                 'latex': r'$\forb{S\,II\]\;\lambda 6732$'}, idx=21)
]


MPL6_specindex_channels = [
    Channel('cn1', formats={'string': 'CN1'}, unit=u.mag, idx=0),
    Channel('cn2', formats={'string': 'CN2'}, unit=u.mag, idx=1),
    Channel('ca4227', formats={'string': 'Ca 4227',
                               'latex': r'Ca\,\lambda 4227'}, unit=u.Angstrom, idx=2),
    Channel('g4300', formats={'string': 'G4300',
                              'latex': r'G\,\lambda 4300'}, unit=u.Angstrom, idx=3),
    Channel('fe4383', formats={'string': 'Fe 4383',
                               'latex': r'Fe\,\lambda 4383'}, unit=u.Angstrom, idx=4),
    Channel('ca4455', formats={'string': 'Ca 4455',
                               'latex': r'Ca\,\lambda 4455'}, unit=u.Angstrom, idx=5),
    Channel('fe4531', formats={'string': 'Fe 4531',
                               'latex': r'Fe\,\lambda 4531'}, unit=u.Angstrom, idx=6),
    Channel('c24668', formats={'string': 'C24668',
                               'latex': r'C2\,\lambda 4668'}, unit=u.Angstrom, idx=7),
    Channel('hb', formats={'string': 'Hb',
                           'latex': r'H\beta'}, unit=u.Angstrom, idx=8),
    Channel('fe5015', formats={'string': 'Fe 5015',
                               'latex': r'Fe\,\lambda 5015'}, unit=u.Angstrom, idx=9),
    Channel('mg1', formats={'string': 'Mg1'}, unit=u.mag, idx=10),
    Channel('mg2', formats={'string': 'Mg2'}, unit=u.mag, idx=11),
    Channel('mgb', formats={'string': 'Mgb'}, unit=u.Angstrom, idx=12),
    Channel('fe5270', formats={'string': 'Fe 5270',
                               'latex': r'Fe\,\lambda 5270'}, unit=u.Angstrom, idx=13),
    Channel('fe5335', formats={'string': 'Fe 5335',
                               'latex': r'Fe\,\lambda 5335'}, unit=u.Angstrom, idx=14),
    Channel('fe5406', formats={'string': 'Fe 5406',
                               'latex': r'Fe\,\lambda 5406'}, unit=u.Angstrom, idx=15),
    Channel('fe5709', formats={'string': 'Fe 5709',
                               'latex': r'Fe\,\lambda 5709'}, unit=u.Angstrom, idx=16),
    Channel('fe5782', formats={'string': 'Fe 5782',
                               'latex': r'Fe\,\lambda 5782'}, unit=u.Angstrom, idx=17),
    Channel('nad', formats={'string': 'NaD'}, unit=u.Angstrom, idx=18),
    Channel('tio1', formats={'string': 'TiO1'}, unit=u.mag, idx=19),
    Channel('tio2', formats={'string': 'TiO2'}, unit=u.mag, idx=20),
    Channel('hdeltaa', formats={'string': 'HDeltaA',
                                'latex': r'H\delta\,A'}, unit=u.Angstrom, idx=21),
    Channel('hgammaa', formats={'string': 'HGammaA',
                                'latex': r'H\gamma\,F'}, unit=u.Angstrom, idx=22),
    Channel('hdeltaf', formats={'string': 'HDeltaA',
                                'latex': r'H\delta\,F'}, unit=u.Angstrom, idx=23),
    Channel('hgammaf', formats={'string': 'HGammaF',
                                'latex': r'H\gamma\,F'}, unit=u.Angstrom, idx=24),
    Channel('cahk', formats={'string': 'CaHK'}, unit=u.Angstrom, idx=25),
    Channel('caii1', formats={'string': 'CaII1'}, unit=u.Angstrom, idx=26),
    Channel('caii2', formats={'string': 'CaII2'}, unit=u.Angstrom, idx=27),
    Channel('caii3', formats={'string': 'CaII3'}, unit=u.Angstrom, idx=28),
    Channel('pa17', formats={'string': 'Pa17'}, unit=u.Angstrom, idx=29),
    Channel('pa14', formats={'string': 'Pa14'}, unit=u.Angstrom, idx=30),
    Channel('pa12', formats={'string': 'Pa12'}, unit=u.Angstrom, idx=31),
    Channel('mgicvd', formats={'string': 'MgICvD'}, unit=u.Angstrom, idx=32),
    Channel('naicvd', formats={'string': 'NaICvD'}, unit=u.Angstrom, idx=33),
    Channel('mgiir', formats={'string': 'MgIIR'}, unit=u.Angstrom, idx=34),
    Channel('fehcvd', formats={'string': 'FeHCvD'}, unit=u.Angstrom, idx=35),
    Channel('nai', formats={'string': 'NaI'}, unit=u.Angstrom, idx=36),
    Channel('btio', formats={'string': 'bTiO'}, unit=u.mag, idx=37),
    Channel('atio', formats={'string': 'aTiO'}, unit=u.mag, idx=38),
    Channel('cah1', formats={'string': 'CaH1'}, unit=u.mag, idx=39),
    Channel('cah2', formats={'string': 'CaH2'}, unit=u.mag, idx=40),
    Channel('naisdss', formats={'string': 'NaISDSS'}, unit=u.Angstrom, idx=41),
    Channel('tio2sdss', formats={'string': 'TiO2SDSS'}, unit=u.Angstrom, idx=42),
    Channel('d4000', formats={'string': 'D4000'}, unit=u.dimensionless_unscaled, idx=43),
    Channel('dn4000', formats={'string': 'Dn4000'}, unit=u.dimensionless_unscaled, idx=44),
    Channel('tiocvd', formats={'string': 'TiOCvD'}, unit=u.dimensionless_unscaled, idx=45)
]


MPL6_binid_channels = [
    Channel('binned_spectra', formats={'string': 'Binned spectra'},
            unit=u.dimensionless_unscaled, idx=0),
    Channel('stellar_continua', formats={'string': 'Stellar continua'},
            unit=u.dimensionless_unscaled, idx=1),
    Channel('em_line_moments', formats={'string': 'Emission line moments'},
            unit=u.dimensionless_unscaled, idx=2),
    Channel('em_line_models', formats={'string': 'Emission line models'},
            unit=u.dimensionless_unscaled, idx=3),
    Channel('spectral_indices', formats={'string': 'Spectral indices'},
            unit=u.dimensionless_unscaled, idx=4)]


binid_properties = MultiChannelProperty('binid', ivar=False, mask=False,
                                        channels=MPL6_binid_channels,
                                        description='Numerical ID for spatial bins.')


MPL6_maps = [
    MultiChannelProperty('spx_skycoo', ivar=False, mask=False,
                         channels=[Channel('on_sky_x', formats={'string': 'On-sky X'}, idx=0),
                                   Channel('on_sky_y', formats={'string': 'On-sky Y'}, idx=1)],
                         unit=u.arcsec,
                         formats={'string': 'Sky coordinates'},
                         description='Offsets of each spaxel from the galaxy center.'),
    MultiChannelProperty('spx_ellcoo', ivar=False, mask=False,
                         channels=[Channel('elliptical_radius',
                                           formats={'string': 'Elliptical radius'},
                                           idx=0, unit=u.arcsec),
                                   Channel('r_re',
                                           formats={'string': 'R/Reff'},
                                           idx=1),
                                   Channel('elliptical_azimuth',
                                           formats={'string': 'Elliptical azimuth'},
                                           idx=2, unit=u.deg)],
                         formats={'string': 'Elliptical coordinates'},
                         description='Elliptical polar coordinates of each spaxel from '
                                     'the galaxy center.'),
    Property('spx_mflux', ivar=True, mask=False,
             unit=u.erg / u.s / (u.cm ** 2) / spaxel_unit, scale=1e-17,
             formats={'string': 'r-band mean flux'},
             description='Mean flux in r-band (5600.1-6750.0 ang).'),
    Property('spx_snr', ivar=False, mask=False,
             formats={'string': 'r-band SNR'},
             description='r-band signal-to-noise ratio per pixel.'),
    binid_properties,
    MultiChannelProperty('bin_lwskycoo', ivar=False, mask=False,
                         channels=[Channel('lum_weighted_on_sky_x',
                                           formats={'string': 'Light-weighted offset X'},
                                           idx=0, unit=u.arcsec),
                                   Channel('lum_weighted_on_sky_y',
                                           formats={'string': 'Light-weighted offset Y'},
                                           idx=1, unit=u.arcsec)],
                         description='Light-weighted offset of each bin from the galaxy center.'),
    MultiChannelProperty('bin_lwellcoo', ivar=False, mask=False,
                         channels=[Channel('lum_weighted_elliptical_radius',
                                           formats={'string': 'Light-weighted radial offset'},
                                           idx=0, unit=u.arcsec),
                                   Channel('r_re',
                                           formats={'string': 'R/REff'},
                                           idx=1),
                                   Channel('lum_weighted_elliptical_azimuth',
                                           formats={'string': 'Light-weighted azimuthal offset'},
                                           idx=2, unit=u.deg)],
                         description='Light-weighted elliptical polar coordinates of each bin '
                                     'from the galaxy center.'),
    Property('bin_area', ivar=False, mask=False,
             unit=u.arcsec ** 2,
             formats={'string': 'Bin area'},
             description='Area of each bin.'),
    Property('bin_farea', ivar=False, mask=False,
             formats={'string': 'Bin fractional area'},
             description='Fractional area that the bin covers for the expected bin '
                         'shape (only relevant for radial binning).'),
    Property('bin_mflux', ivar=True, mask=True,
             unit=u.erg / u.s / (u.cm ** 2) / spaxel_unit, scale=1e-17,
             formats={'string': 'r-band binned spectra mean flux'},
             description='Mean flux in the r-band for the binned spectra.'),
    Property('bin_snr', ivar=False, mask=False,
             formats={'string': 'Bin SNR'},
             description='r-band signal-to-noise ratio per pixel in the binned spectra.'),
    Property('stellar_vel', ivar=True, mask=True,
             unit=u.km / u.s,
             formats={'string': 'Stellar velocity'},
             description='Stellar velocity relative to NSA redshift.'),
    Property('stellar_sigma', ivar=True, mask=True,
             unit=u.km / u.s,
             formats={'string': 'Stellar velocity dispersion', 'latex': r'Stellar $\sigma$'},
             description='Stellar velocity dispersion (must be corrected using '
                         'STELLAR_SIGMACORR)'),
    Property('stellar_sigmacorr', ivar=False, mask=False,
             unit=u.km / u.s,
             formats={'string': 'Stellar sigma correction',
                      'latex': r'Stellar $\sigma$ correction'},
             description='Quadrature correction for STELLAR_SIGMA to obtain the '
                         'astrophysical velocity dispersion.)'),
    MultiChannelProperty('stellar_cont_fresid', ivar=False, mask=False,
                         channels=[Channel('68th_percentile',
                                           formats={'string': '68th percentile',
                                                    'latex': r'68^{th} percentile'}, idx=0),
                                   Channel('99th_percentile',
                                           formats={'string': '99th percentile',
                                                    'latex': r'99^{th} percentile'}, idx=1)],
                         formats={'string': 'Fractional residual growth'},
                         description='68%% and 99%% growth of the fractional residuals between '
                                     'the model and data.'),
    Property('stellar_cont_rchi2', ivar=False, mask=False,
             formats={'string': 'Stellar continuum reduced chi-square',
                      'latex': r'Stellar\ continuum\ reduced\ \chi^2'},
             description='Reduced chi-square of the stellar continuum fit.'),
    MultiChannelProperty('emline_sflux', ivar=True, mask=True,
                         channels=[oiid_channel] + MPL6_emline_channels,
                         formats={'string': 'Emission line summed flux'},
                         unit=u.erg / u.s / (u.cm ** 2) / spaxel_unit, scale=1e-17,
                         binid=binid_properties[3],
                         description='Non-parametric summed flux for emission lines.'),
    MultiChannelProperty('emline_sew', ivar=True, mask=True,
                         channels=[oiid_channel] + MPL6_emline_channels,
                         formats={'string': 'Emission line EW'},
                         unit=u.Angstrom,
                         binid=binid_properties[3],
                         description='Emission line non-parametric equivalent '
                                     'widths measurements.'),
    MultiChannelProperty('emline_gflux', ivar=True, mask=True,
                         channels=[oii_channel] + MPL6_emline_channels,
                         formats={'string': 'Emission line Gaussian flux'},
                         unit=u.erg / u.s / (u.cm ** 2) / spaxel_unit, scale=1e-17,
                         binid=binid_properties[3],
                         description='Gaussian profile integrated flux for emission lines.'),
    MultiChannelProperty('emline_gvel', ivar=True, mask=True,
                         channels=[oii_channel] + MPL6_emline_channels,
                         formats={'string': 'Emission line Gaussian velocity'},
                         unit=u.km / u.s,
                         binid=binid_properties[3],
                         description='Gaussian profile velocity for emission lines.'),
    MultiChannelProperty('emline_gew', ivar=True, mask=True,
                         channels=[oii_channel] + MPL6_emline_channels,
                         formats={'string': 'Emission line Gaussian EW'},
                         unit=u.Angstrom,
                         binid=binid_properties[3],
                         description='Gaussian-fitted equivalent widths measurements '
                                     '(based on EMLINE_GFLUX).'),
    MultiChannelProperty('emline_gsigma', ivar=True, mask=True,
                         channels=[oii_channel] + MPL6_emline_channels,
                         formats={'string': 'Emission line Gaussian sigma',
                                  'latex': r'Emission line Gaussian $\sigma$'},
                         unit=u.km / u.s,
                         binid=binid_properties[3],
                         description='Gaussian profile velocity dispersion for emission lines; '
                                     'must be corrected using EMLINE_INSTSIGMA.'),
    MultiChannelProperty('emline_instsigma', ivar=False, mask=False,
                         channels=[oii_channel] + MPL6_emline_channels,
                         formats={'string': 'Emission line instrumental sigma',
                                  'latex': r'Emission line instrumental $\sigma$'},
                         unit=u.km / u.s,
                         binid=binid_properties[3],
                         description='Instrumental dispersion at the fitted line center.'),
    MultiChannelProperty('emline_tplsigma', ivar=False, mask=False,
                         channels=[oii_channel] + MPL6_emline_channels,
                         formats={'string': 'Emission line template instrumental sigma',
                                  'latex': r'Emission line template instrumental $\sigma$'},
                         unit=u.km / u.s,
                         binid=binid_properties[3],
                         description='The dispersion of each emission line used in '
                                     'the template spectra'),
    MultiChannelProperty('specindex', ivar=True, mask=True,
                         channels=MPL6_specindex_channels,
                         formats={'string': 'Spectral index'},
                         description='Measurements of spectral indices.'),
    MultiChannelProperty('specindex_corr', ivar=False, mask=False,
                         channels=MPL6_specindex_channels,
                         formats={'string': 'Spectral index sigma correction',
                                  'latex': r'Spectral index $\sigma$ correction'},
                         description='Velocity dispersion corrections for the '
                                     'spectral index measurements '
                                     '(can be ignored for D4000, Dn4000).')
]


MPL6_models = [
    Model('binned_flux', 'FLUX', 'WAVE', extension_ivar='IVAR',
          extension_mask='MASK', unit=u.erg / u.s / (u.cm ** 2) / spaxel_unit,
          scale=1e-17, formats={'string': 'Binned flux'},
          description='Flux of the binned spectra',
          binid=binid_properties[0]),
    Model('full_fit', 'MODEL', 'WAVE', extension_ivar=None,
          extension_mask='MASK', unit=u.erg / u.s / (u.cm ** 2) / spaxel_unit,
          scale=1e-17, formats={'string': 'Best fitting model'},
          description='The best fitting model spectra (sum of the fitted '
                      'continuum and emission-line models)',
          binid=binid_properties[0]),
    Model('emline_fit', 'EMLINE', 'WAVE', extension_ivar=None,
          extension_mask='EMLINE_MASK',
          unit=u.erg / u.s / (u.cm ** 2) / spaxel_unit,
          scale=1e-17, formats={'string': 'Emission line model spectrum'},
          description='The model spectrum with only the emission lines.',
          binid=binid_properties[3]),
    Model('emline_base_fit', 'EMLINE_BASE', 'WAVE', extension_ivar=None,
          extension_mask='EMLINE_MASK',
          unit=u.erg / u.s / (u.cm ** 2) / spaxel_unit,
          scale=1e-17, formats={'string': 'Emission line baseline fit'},
          description='The model of the constant baseline fitted beneath the '
                      'emission lines.',
          binid=binid_properties[3])
]


# MPL-6 DapDataModel goes here
MPL6 = DAPDataModel('2.1.3', aliases=['MPL-6', 'MPL6'],
                    bintypes=[SPX, HYB10, VOR10, ALL, NRE],
                    db_only=[SPX, HYB10],
                    templates=[GAU_MILESHC],
                    properties=MPL6_maps,
                    models=MPL6_models,
                    bitmasks=get_maskbits('MPL-6'),
                    default_bintype='SPX',
                    default_template='GAU-MILESHC',
                    property_table='SpaxelProp6',
                    default_binid=binid_properties[0],
                    default_mapmask=['NOCOV', 'UNRELIABLE', 'DONOTUSE'],
                    qual_flag='DAPQUAL')
