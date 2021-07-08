from __future__ import print_function, division, absolute_import

from .base import VACMixIn, VACTarget

import numpy as np
import matplotlib.patches as patches
import scipy.linalg as sl
import json
import marvin.utils.dap.bpt as bpt
import marvin

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import distance_matrix
from marvin.tools.quantities.spectrum import Spectrum
from marvin import log

LUT = {7: 3, 19: 5, 37: 7, 61: 9, 91: 11, 127: 13}
spaxel_grid = {7: 24, 19: 34, 37: 44, 61: 54, 91: 64, 127: 74}


def convert_json(table, column_name):
    # this unpacks the json column of a table
    new_col = [json.loads(i) for i in table[column_name]]
    table.rename_column(column_name, '{0}_string'.format(column_name))
    table['{0}_list'.format(column_name)] = new_col


def non_blank(table, *column_name):
    for cdx, c in enumerate(column_name):
        if cdx == 0:
            non_blank = np.array([len(i) > 0 for i in table[c]])
        else:
            non_blank = non_blank | np.array([len(i) > 0 for i in table[c]])
    return non_blank.sum()


def cov_to_ellipse(cov, pos, nstd=1, **kwargs):
    eigvec, eigval, V = sl.svd(cov, full_matrices=False)
    # the angle the first eigenvector makes with the x-axis
    theta = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
    # full width and height of ellipse, not radius
    # the eigenvalues are the variance along the eigenvectors
    width, height = 2 * nstd * np.sqrt(eigval)
    return patches.Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)


class Suppressor(object):
    # A class to mute the output of a function
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout

    def write(self, x):
        pass


class GZ3DVAC(VACMixIn):

    name = 'gz3d'
    description = 'Return object for working with Galaxy Zoo 3D data masks'
    version = {'DR17': 'v4_0_0', 'MPL-11': 'v4_0_0'}

    include = (marvin.tools.cube.Cube, marvin.tools.maps.Maps)

    def set_summary_file(self, release):
        ''' Sets the path to the GalaxyZoo3D summary file '''
        self.path_params = {'ver': self.version[release]}

        self.summary_file = self.get_path('mangagz3dmetadata', path_params=self.path_params)
        self.center_summary_file = self.get_path('mangagz3dcenters', path_params=self.path_params)
        self.stars_summary_file = self.get_path('mangagz3dstars', path_params=self.path_params)

    def get_target(self, parent_object):
        mangaid = parent_object.mangaid

        if parent_object.__class__ == marvin.tools.cube.Cube:
            cube = parent_object
            maps = parent_object.getMaps()
        else:
            cube = parent_object.getCube()
            maps = parent_object

        if not self.file_exists(self.summary_file):
            self.summary_file = self.download_vac('mangagz3dmetadata', path_params=self.path_params)
            self.center_summary_file = self.download_vac('mangagz3dcenters', path_params=self.path_params)
            self.stars_summary_file = self.download_vac('mangagz3dstars', path_params=self.path_params)

        summary_table = Table.read(self.summary_file, hdu=1)
        # Table adds extra spaces to short strings, these need to be stripped off
        gz3d_mangaids = np.array([mid.strip() for mid in summary_table['MANGAID']])

        idx = gz3d_mangaids == mangaid
        if idx.sum() > 0:
            file_name = summary_table[idx]['file_name'][0].strip()

            self.path_params.update(file_name=file_name)
            self.gz3d_filename = self.get_path('mangagz3d', path_params=self.path_params)

            if not self.file_exists(self.gz3d_filename):
                self.gz3d_filename = self.download_vac('mangagz3d', path_params=self.path_params)

            return GZ3DTarget(self.gz3d_filename, cube, maps)

        log.info('There is no GZ3D data for this mangaid: {0}'.format(mangaid))
        return None


class GZ3DTarget(object):
    def __init__(self, filename, cube, maps):
        # get the subject id from the filename
        self.subject_id = filename.split('/')[-1].split('_')[-1].split('.')[0]
        # read in the fits file
        self.hdulist = fits.open(filename)
        # grab the wcs
        self.wcs = WCS(self.hdulist[1].header)
        self.process_images()
        # read in metadata
        self.metadata = Table(self.hdulist[5].data)
        self.ifu_size = int(self.metadata['IFUDESIGNSIZE'][0])
        self.process_clusters()
        self.process_clusters_classifications()
        self.process_spiral_classifications()
        self.process_bar_classifications()
        self.cube = cube
        self.maps = maps
        self.spaxel_masks = False
        self.mean_bar = None
        self.mean_spiral = None
        self.mean_center = None
        self.mean_not_bar = None
        self.mean_not_spiral = None
        self.mean_not_center = None
        self.log_oiii_hb = None
        self.log_nii_ha = None
        self.log_sii_ha = None
        self.log_oi_ha = None
        self.dis = None

    def process_images(self):
        # read in images
        self.image = self.hdulist[0].data
        self.center_mask = self.hdulist[1].data
        self.star_mask = self.hdulist[2].data
        self.spiral_mask = self.hdulist[3].data
        self.bar_mask = self.hdulist[4].data

    def process_clusters(self):
        # read in center clusters
        self.center_clusters = Table(self.hdulist[6].data)
        self.num_centers = len(self.center_clusters)
        # read in star clusters
        self.star_clusters = Table(self.hdulist[7].data)
        self.num_stars = len(self.star_clusters)

    def process_clusters_classifications(self):
        # read in center and star classifications
        self.center_star_classifications = Table(self.hdulist[8].data)
        self.num_center_star_classifications = len(self.center_star_classifications)
        convert_json(self.center_star_classifications, 'center_points')
        convert_json(self.center_star_classifications, 'star_points')
        self.num_center_star_classifications_non_blank = non_blank(self.center_star_classifications, 'center_points_list', 'star_points_list')

    def process_spiral_classifications(self):
        # read in spiral classifications
        self.spiral_classifications = Table(self.hdulist[9].data)
        self.num_spiral_classifications = len(self.spiral_classifications)
        convert_json(self.spiral_classifications, 'spiral_paths')
        self.num_spiral_classifications_non_blank = non_blank(self.spiral_classifications, 'spiral_paths_list')

    def process_bar_classifications(self):
        # read in bar classifications
        self.bar_classifications = Table(self.hdulist[10].data)
        self.num_bar_classifications = len(self.bar_classifications)
        convert_json(self.bar_classifications, 'bar_paths')
        self.num_bar_classifications_non_blank = non_blank(self.bar_classifications, 'bar_paths_list')

    def center_in_pix(self):
        return self.wcs.wcs_world2pix(np.array([[self.metadata['ra'][0], self.metadata['dec'][0]]]), 1)[0]

    def get_hexagon(self, correct_hex=False, edgecolor='magenta'):
        # the spacing should be ~0.5 arcsec not 0, and it should not be rotated by np.sqrt(3) / 2
        if correct_hex:
            # each hex has a total diameter of 2.5 arcsec on the sky (only 2 of it is a fiber)
            diameter = 2.5 / 0.099
            # the radius for mpl is from the center to each vertex, not center to side
            r = LUT[self.ifu_size] * diameter / 2
        else:
            # this was me being wrong about all the hexagon params
            # these hexagons are about 0.7 times too small (2 * np.sqrt(3) / 5 to be exact)
            diameter = 2.0 / 0.099
            r = LUT[self.ifu_size] * diameter * np.sqrt(3) / 4
        c = self.center_in_pix()
        return patches.RegularPolygon(c, 6, r, fill=False, orientation=np.deg2rad(30), edgecolor=edgecolor, linewidth=0.8)

    def _get_ellipse_list(self, table):
        ellip_list = []
        for idx in range(len(table)):
            pos = np.array([table['x'][idx], table['y'][idx]])
            cov = np.array([[table['var_x'][idx], table['var_x_y'][idx]], [table['var_x_y'][idx], table['var_y'][idx]]])
            ellip_list.append(cov_to_ellipse(cov, pos, nstd=2, edgecolor='k', facecolor='none', lw=1))
        return ellip_list

    def get_center_ellipse_list(self):
        return self._get_ellipse_list(self.center_clusters)

    def get_star_ellipse_list(self):
        return self._get_ellipse_list(self.star_clusters)

    def _get_spaxel_grid_xy(self, include_edges=False, grid_size=None):
        if grid_size is None:
            grid_size = self.cube.data['FLUX'].data.shape[1:]
        one_grid = 0.5 / 0.099
        c = self.center_in_pix()
        grid_y = np.arange(grid_size[0] + include_edges) * one_grid
        grid_x = np.arange(grid_size[1] + include_edges) * one_grid
        grid_y = grid_y - np.median(grid_y) + c[0]
        grid_x = grid_x - np.median(grid_x) + c[1]
        return grid_x, grid_y

    def get_spaxel_grid(self, grid_size=None):
        grid_x, grid_y = self._get_spaxel_grid_xy(include_edges=True, grid_size=grid_size)
        v_line_x = np.array(zip(grid_x, grid_x)).T
        v_line_y = np.array([(grid_y[0], grid_y[-1])]).T
        h_line_x = np.array([(grid_x[0], grid_x[-1])]).T
        h_line_y = np.array(zip(grid_y, grid_y)).T
        return [(v_line_x, v_line_y), (h_line_x, h_line_y)]

    def _get_spaxel_mask(self, mask, grid_size=None):
        # assumes a 0.5 arcsec grid centered on the ifu's ra and dec
        # use a Bivariate spline approximation to resample mask to the spaxel grid
        xx = np.arange(mask.shape[1])
        yy = np.arange(mask.shape[0])
        s = RectBivariateSpline(xx, yy, mask)
        grid_x, grid_y = self._get_spaxel_grid_xy(grid_size=grid_size)
        # flip the output mask so the origin is the lower left of the image
        s_mask = np.flipud(s(grid_x, grid_y))
        # zero out small values
        s_mask[s_mask < 0.5] = 0
        return s_mask

    def make_all_spaxel_masks(self, grid_size=None):
        if not self.spaxel_masks:
            self.center_mask_spaxel = self._get_spaxel_mask(self.center_mask, grid_size=grid_size)
            self.star_mask_spaxel = self._get_spaxel_mask(self.star_mask, grid_size=grid_size)
            self.spiral_mask_spaxel = self._get_spaxel_mask(self.spiral_mask, grid_size=grid_size)
            self.bar_mask_spaxel = self._get_spaxel_mask(self.bar_mask, grid_size=grid_size)
            self.other_mask_spaxel = (self.spiral_mask_spaxel == 0) & (self.bar_mask_spaxel == 0) & (self.center_mask_spaxel == 0)
            self.spaxel_masks = True

    def _stack_spectra(self, mask_name, inv=False):
        mask = getattr(self, mask_name)
        if inv:
            mask = mask.max() - mask
        mdx = np.where(mask > 0)
        if len(mdx[0] > 0):
            weights = mask[mdx]
            spaxel_index = np.array(mdx.nonzero()).T

            spectra = [s.flux.copy() for s in self.cube[mdx.nonzero()]]

            # only keep spectra inside the IFU
            in_ifu = np.array([not any(2**0 & s.mask) for s in spectra])
            if in_ifu.sum() == 0:
                return None

            spectra = [spectra[i] for i in in_ifu.nonzero()[0]]
            spaxel_index = spaxel_index[in_ifu]
            weights = weights[in_ifu]
            weights_total = weights.sum()

            if len(spectra) == 1:
                return spectra[0]

            # we need to handle covariance between spaxels when calculating
            # uncertainties. We follow Westfall et al. 2019's method based in
            # distance between spaxels

            d = distance_matrix(spaxel_index, spaxel_index) / 1.92
            roh = np.exp(-0.5 * d**2)

            flux = np.array([sp.value for sp in spectra])
            # the weighted mean
            mean = (flux * weights[:, None]).sum(axis=0) / weights_total

            sigma = np.array([sp.error.value for sp in spectra])
            sigma_weights = weights[:, None] * sigma
            sigma_outer = sigma_weights[None, :, :] * sigma_weights[:, None, :]
            roh_sigma = roh[:, :, None] * sigma_outer
            ivar = (weights_total**2) / roh_sigma.sum(axis=(0, 1))

            mask = spectra[0].mask
            for sp in spectra[1:]:
                mask |= sp.mask

            return Spectrum(
                mean,
                unit=spectra[0].unit,
                wavelength=spectra[0].wavelength,
                wavelength_unit=spectra[0].wavelength.unit,
                pixmask_flag=spectra[0].pixmask_flag,
                ivar=ivar,
                mask=mask
            )

        return None

    def get_mean_spectra(self, inv=False):
        self.make_all_spaxel_masks()
        self.mean_bar = self._stack_spectra('bar_mask_spaxel')
        self.mean_spiral = self._stack_spectra('spiral_mask_spaxel')
        self.mean_center = self._stack_spectra('center_mask_spaxel')
        if inv:
            self.mean_not_bar = self._stack_spectra('bar_mask_spaxel', inv=True)
            self.mean_not_spiral = self._stack_spectra('spiral_mask_spaxel', inv=True)
            self.mean_not_center = self._stack_spectra('center_mask_spaxel', inv=True)

    def get_bpt(self, snr_min=3, oi_sf=False):
        # Gets the necessary emission line maps
        oiii = bpt.get_masked(self.maps, 'oiii_5008', snr=bpt.get_snr(snr_min, 'oiii'))
        nii = bpt.get_masked(self.maps, 'nii_6585', snr=bpt.get_snr(snr_min, 'nii'))
        ha = bpt.get_masked(self.maps, 'ha_6564', snr=bpt.get_snr(snr_min, 'ha'))
        hb = bpt.get_masked(self.maps, 'hb_4862', snr=bpt.get_snr(snr_min, 'hb'))
        sii = bpt.get_masked(self.maps, 'sii_6718', snr=bpt.get_snr(snr_min, 'sii'))
        oi = bpt.get_masked(self.maps, 'oi_6302', snr=bpt.get_snr(snr_min, 'oi'))
        self.log_oiii_hb = np.ma.log10(oiii / hb)
        self.log_nii_ha = np.ma.log10(nii / ha)
        self.log_sii_ha = np.ma.log10(sii / ha)
        self.log_oi_ha = np.ma.log10(oi / ha)
        sf_mask_nii = ((self.log_oiii_hb < bpt.kewley_sf_nii(self.log_nii_ha)) & (self.log_nii_ha < 0.05)).filled(False)
        sf_mask_sii = ((self.log_oiii_hb < bpt.kewley_sf_sii(self.log_sii_ha)) & (self.log_sii_ha < 0.32)).filled(False)
        sf_mask_oi = ((self.log_oiii_hb < bpt.kewley_sf_oi(self.log_oi_ha)) & (self.log_oi_ha < -0.59)).filled(False)
        if oi_sf:
            self.sf_mask = sf_mask_nii & sf_mask_sii & sf_mask_oi
        else:
            self.sf_mask = sf_mask_nii & sf_mask_sii

    def bpt_in_mask(self, mask_name, bpt_name, factor=1.2):
        self.get_distance()
        mask = getattr(self, mask_name)
        bpt_data = getattr(self, bpt_name)
        outside = self.log_oiii_hb.mask | bpt_data.mask
        mdx = np.where((mask > 0) & (~outside))
        order = np.argsort(self.dis[mdx])
        dis_max = 1
        if (~outside).sum() > 0:
            dis_max = self.dis[~outside].max() * factor
        return bpt_data[mdx][order], self.log_oiii_hb[mdx][order], self.dis[mdx][order] / dis_max

    def get_distance(self):
        if self.dis is None:
            self.make_all_spaxel_masks()
            cdx = np.unravel_index(self.center_mask_spaxel.argmax(), self.center_mask_spaxel.shape)
            self.dis = np.zeros_like(self.center_mask_spaxel)
            for yy in range(self.dis.shape[0]):
                for xx in range(self.dis.shape[1]):
                    self.dis[yy, xx] = np.linalg.norm([yy - cdx[0], xx - cdx[1]])

    def close(self):
        self.hdulist.close()

    def __str__(self):
        return '\n'.join([
            'Subject info:',
            '    subject id: {0}'.format(self.subject_id),
            '    manga id: {0}'.format(self.metadata['MANGAID'][0]),
            '    ra: {0}'.format(self.metadata['ra'][0]),
            '    dec: {0}'.format(self.metadata['dec'][0]),
            '    ifu size: {0}'.format(self.ifu_size),
            'Classification counts:',
            '    {0} center/star, {1} non_blank'.format(self.num_center_star_classifications, self.num_center_star_classifications_non_blank),
            '    {0} spiral, {1} non_blank'.format(self.num_spiral_classifications, self.num_spiral_classifications_non_blank),
            '    {0} bar, {1} non_blank'.format(self.num_bar_classifications, self.num_bar_classifications_non_blank),
            'Cluster counts:',
            '    {0} center(s)'.format(self.num_centers),
            '    {0} star(s)'.format(self.num_stars)
        ])
