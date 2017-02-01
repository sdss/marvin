#!/usr/bin/env python
# encoding: utf-8
#
# bpt.py
#
# Created by José Sánchez-Gallego on 19 Jan 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid


def get_masked(maps, emline):
    """Convenience function to get masked arrays without negative values."""

    gflux = maps['emline_gflux_' + emline].masked
    gflux.mask |= (gflux.data <= 0)

    return gflux


def _get_kewley06_axes():
    """Creates custom axes for displaying Kewley06 plots."""

    fig = plt.figure(1, (8.5, 10))
    fig.clf()

    plt.subplots_adjust(top=0.99, bottom=0.08, hspace=0.01)

    grid_bpt = ImageGrid(fig, 211,
                         nrows_ncols=(1, 3),
                         direction='row',
                         axes_pad=0.,
                         add_all=True,
                         label_mode='L',
                         share_all=False)

    gal_bpt = ImageGrid(fig, 212, nrows_ncols=(1, 1))

    xx_sf_nii = np.linspace(-1.281, 0.045, 1e4)
    xx_sf_sii = np.linspace(-2, 0.315, 1e4)
    xx_sf_oi = np.linspace(-2.5, -0.7, 1e4)

    xx_comp_nii = np.linspace(-2, 0.4, 1e4)

    xx_agn_sii = np.array([-0.308, 1.0])
    xx_agn_oi = np.array([-1.12, 0.5])

    grid_bpt[0].plot(xx_sf_nii, sf_nii(xx_sf_nii), 'b--', zorder=90)
    grid_bpt[1].plot(xx_sf_sii, sf_sii(xx_sf_sii), 'r-', zorder=90)
    grid_bpt[2].plot(xx_sf_oi, sf_oi(xx_sf_oi), 'r-', zorder=90)

    grid_bpt[0].plot(xx_comp_nii, comp_nii(xx_comp_nii), 'r-', zorder=90)

    grid_bpt[1].plot(xx_agn_sii, agn_sii(xx_agn_sii), 'b-', zorder=80)
    grid_bpt[2].plot(xx_agn_oi, agn_oi(xx_agn_oi), 'b-', zorder=80)

    grid_bpt[0].text(-1, -0.5, 'SF', ha='center', fontsize=12, zorder=100, color='c')
    grid_bpt[0].text(0, 1, 'AGN', ha='left', fontsize=12, zorder=100)
    grid_bpt[0].text(-0.08, -1.2, 'Comp', ha='left', fontsize=12, zorder=100, color='g')

    grid_bpt[1].text(-1.2, -0.5, 'SF', ha='center', fontsize=12, zorder=100)
    grid_bpt[1].text(-1, 1.2, 'Seyfert', ha='left', fontsize=12, zorder=100, color='r')
    grid_bpt[1].text(0.3, -1, 'LINER', ha='left', fontsize=12, zorder=100, color='m')

    grid_bpt[2].text(-2, -0.5, 'SF', ha='center', fontsize=12, zorder=100)
    grid_bpt[2].text(-1.5, 1, 'Seyfert', ha='left', fontsize=12, zorder=100)
    grid_bpt[2].text(-0.1, -1, 'LINER', ha='right', fontsize=12, zorder=100)

    xtick_limits = ((-2, 1), (-1.5, 1), (-2.5, 0.5))

    for ii in [0, 1, 2]:

        grid_bpt[ii].get_xaxis().set_tick_params(direction='in')
        grid_bpt[ii].get_yaxis().set_tick_params(direction='in')

        grid_bpt[ii].set_xticks(np.arange(xtick_limits[ii][0], xtick_limits[ii][1] + 0.5, 0.5))
        grid_bpt[ii].set_xticks(np.arange(xtick_limits[ii][0],
                                          xtick_limits[ii][1] + 0.1, 0.1), minor=True)
        grid_bpt[ii].set_yticks(np.arange(-1.5, 2.0, 0.5))
        grid_bpt[ii].set_yticks(np.arange(-1.5, 1.6, 0.1), minor=True)

        grid_bpt[ii].grid(which='minor', alpha=0.2)
        grid_bpt[ii].grid(which='major', alpha=0.5)

        grid_bpt[ii].set_xlim(xtick_limits[ii][0], xtick_limits[ii][1])
        grid_bpt[ii].set_ylim(-1.5, 1.5)

        grid_bpt[ii].spines['top'].set_visible(True)

        if ii in [1, 2]:
            grid_bpt[ii].get_xticklabels()[0].set_visible(False)

    grid_bpt[0].set_ylabel(r'log([OIII]/H$\beta$)')

    grid_bpt[0].set_xlabel(r'log([NII]/H$\alpha$)')
    grid_bpt[1].set_xlabel(r'log([SII]/H$\alpha$)')
    grid_bpt[2].set_xlabel(r'log([OI]/H$\alpha$)')

    gal_bpt[0].grid(False)

    return fig, grid_bpt, gal_bpt[0]


def sf_nii(log_nii_ha):
    """Star forming classification line for log([NII]/Ha)."""
    return 0.61 / (log_nii_ha - 0.05) + 1.3


def sf_sii(log_sii_ha):
    """Star forming classification line for log([SII]/Ha)."""
    return 0.72 / (log_sii_ha - 0.32) + 1.3


def sf_oi(log_oi_ha):
    """Star forming classification line for log([OI]/Ha)."""
    return 0.73 / (log_oi_ha + 0.59) + 1.33


def comp_nii(log_nii_ha):
    """Composite classification line for log([NII]/Ha)."""
    return 0.61 / (log_nii_ha - 0.47) + 1.19


def agn_sii(log_sii_ha):
    """Seyfert/LINER classification line for log([SII]/Ha)."""
    return 1.89 * log_sii_ha + 0.76


def agn_oi(log_oi_ha):
    """Seyfert/LINER classification line for log([OI]/Ha)."""
    return 1.18 * log_oi_ha + 1.30


def bpt_kewley06(maps, mode='strict'):
    """Returns ionisation regions, making use of the boundaries defined in Kewley+06."""

    oiii = get_masked(maps, 'oiii_5008')
    nii = get_masked(maps, 'nii_6585')
    ha = get_masked(maps, 'ha_6564')
    hb = get_masked(maps, 'hb_4862')
    sii = get_masked(maps, 'sii_6718')
    oi = get_masked(maps, 'oi_6302')

    log_oiii_hb = np.ma.log10(oiii / hb)
    log_nii_ha = np.ma.log10(nii / ha)
    log_sii_ha = np.ma.log10(sii / ha)
    log_oi_ha = np.ma.log10(oi / ha)

    sf_mask = (((log_oiii_hb < sf_nii(log_nii_ha)) & (log_nii_ha < 0.05)).filled(False) &
               ((log_oiii_hb < sf_sii(log_sii_ha)) & (log_sii_ha < 0.32)).filled(False) &
               ((log_oiii_hb < sf_oi(log_oi_ha)) & (log_oi_ha < -0.59)).filled(False))

    comp_mask = (((log_oiii_hb > sf_nii(log_nii_ha)) & (log_nii_ha < 0.05)).filled(False) &
                 ((log_oiii_hb < comp_nii(log_nii_ha)) & (log_nii_ha < 0.465)).filled(False) &
                 ((log_oiii_hb < sf_sii(log_sii_ha)) & (log_sii_ha < 0.32)).filled(False) &
                 ((log_oiii_hb < sf_oi(log_oi_ha)) & (log_oi_ha < -0.59)).filled(False))

    agn_mask = (((log_oiii_hb > comp_nii(log_nii_ha)) | (log_nii_ha > 0.465)).filled(False) &
                ((log_oiii_hb > sf_sii(log_sii_ha)) | (log_sii_ha > 0.32)).filled(False) &
                ((log_oiii_hb > sf_oi(log_oi_ha)) | (log_oi_ha > -0.59)).filled(False))

    seyfert_mask = (agn_mask & (1.89 * log_sii_ha + 0.76 < log_oiii_hb).filled(False) &
                               (1.18 * log_oi_ha + 1.30 < log_oiii_hb).filled(False))

    liner_mask = (agn_mask & (1.89 * log_sii_ha + 0.76 > log_oiii_hb).filled(False) &
                             (1.18 * log_oi_ha + 1.30 > log_oiii_hb).filled(False))

    invalid_mask = ha.mask & oiii.mask & nii.mask & hb.mask & sii.mask & oi.mask
    ambiguous_mask = ~(sf_mask | comp_mask | seyfert_mask | liner_mask) & ~invalid_mask

    fig, grid_bpt, gal_bpt = _get_kewley06_axes()

    sf_kwargs = {'marker': 's', 's': 10, 'color': 'c', 'zorder': 50}
    grid_bpt[0].scatter(log_nii_ha[sf_mask], log_oiii_hb[sf_mask], **sf_kwargs)
    grid_bpt[1].scatter(log_sii_ha[sf_mask], log_oiii_hb[sf_mask], **sf_kwargs)
    grid_bpt[2].scatter(log_oi_ha[sf_mask], log_oiii_hb[sf_mask], **sf_kwargs)

    comp_kwargs = {'marker': 's', 's': 10, 'color': 'g', 'zorder': 45}
    grid_bpt[0].scatter(log_nii_ha[comp_mask], log_oiii_hb[comp_mask], **comp_kwargs)
    grid_bpt[1].scatter(log_sii_ha[comp_mask], log_oiii_hb[comp_mask], **comp_kwargs)
    grid_bpt[2].scatter(log_oi_ha[comp_mask], log_oiii_hb[comp_mask], **comp_kwargs)

    seyfert_kwargs = {'marker': 's', 's': 10, 'color': 'r', 'zorder': 40}
    grid_bpt[0].scatter(log_nii_ha[seyfert_mask], log_oiii_hb[seyfert_mask], **seyfert_kwargs)
    grid_bpt[1].scatter(log_sii_ha[seyfert_mask], log_oiii_hb[seyfert_mask], **seyfert_kwargs)
    grid_bpt[2].scatter(log_oi_ha[seyfert_mask], log_oiii_hb[seyfert_mask], **seyfert_kwargs)

    liner_kwargs = {'marker': 's', 's': 10, 'color': 'm', 'zorder': 35}
    grid_bpt[0].scatter(log_nii_ha[liner_mask], log_oiii_hb[liner_mask], **liner_kwargs)
    grid_bpt[1].scatter(log_sii_ha[liner_mask], log_oiii_hb[liner_mask], **liner_kwargs)
    grid_bpt[2].scatter(log_oi_ha[liner_mask], log_oiii_hb[liner_mask], **liner_kwargs)

    amb_kwargs = {'marker': 's', 's': 10, 'color': '0.6', 'zorder': 30}
    grid_bpt[0].scatter(log_nii_ha[ambiguous_mask], log_oiii_hb[ambiguous_mask], **amb_kwargs)
    grid_bpt[1].scatter(log_sii_ha[ambiguous_mask], log_oiii_hb[ambiguous_mask], **amb_kwargs)
    grid_bpt[2].scatter(log_oi_ha[ambiguous_mask], log_oiii_hb[ambiguous_mask], **amb_kwargs)

    gal_rgb = np.zeros((ha.shape[0], ha.shape[1], 3), dtype=np.uint8)
    for ii in [1, 2]:
        gal_rgb[:, :, ii][sf_mask] = 255

    gal_rgb[:, :, 1][comp_mask] = 128

    gal_rgb[:, :, 0][seyfert_mask] = 255

    gal_rgb[:, :, 0][liner_mask] = 255
    gal_rgb[:, :, 2][liner_mask] = 255

    for ii in [0, 1, 2]:
        gal_rgb[:, :, ii][invalid_mask] = 255
        gal_rgb[:, :, ii][ambiguous_mask] = 169

    gal_bpt.imshow(gal_rgb, origin='lower', aspect='auto')

    gal_bpt.set_xlim(0, ha.shape[1] - 1)
    gal_bpt.set_ylim(0, ha.shape[0] - 1)
    gal_bpt.set_xlabel('x [spaxels]')
    gal_bpt.set_ylabel('y [spaxels]')

    return fig
