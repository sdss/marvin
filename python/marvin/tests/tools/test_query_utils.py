from __future__ import print_function, division, absolute_import

import unittest
from marvin import config
from marvin.tools.query import query_utils


class ExpandFilterAliasesTestCase(unittest.TestCase):
    """Tests from 'query_utils.py'."""

    @classmethod
    def setUpClass(cls):
        config.use_sentry = False
        config.add_github_message = False

    def test_D4000_value_gt_1(self):
        desired = ("specindex_type.name == 'D4000' and specindex.value > 1")
        parsed_filter = ['D4000', '>', '1']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_D4000_ivar_ne_0(self):
        desired = ("specindex_type.name == 'D4000' and specindex.ivar != 0")
        parsed_filter = ['D4000_ivar', '!=', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_D4000_mask_gt_0(self):
        desired = ("specindex_type.name == 'D4000' and specindex.mask > 0")
        parsed_filter = ['D4000_mask', '>', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_CaII0p86B_value_gt_0(self):
        desired = ("specindex_type.name == 'CaII0p86B' and specindex.value > 0")
        parsed_filter = ['CaII0p86B', '>', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_CaII0p86B_mask_gt_0(self):
        desired = ("specindex_type.name == 'CaII0p86B' and specindex.mask > 0")
        parsed_filter = ['CaII0p86B_mask', '>', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_caii0p86b_mask_gt_1(self):
        desired = ("specindex_type.name == 'CaII0p86B' and specindex.mask > 1")
        parsed_filter = ['caii0p86b_mask', '>', '1']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_Hb_value_gt_1(self):
        desired = ("specindex_type.name == 'Hb' and specindex.value > 1")
        parsed_filter = ['Hb', '>', '1']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_st_vel_lt_300(self):
        desired = ("stellar_kin_parameter.name == 'VEL' and stellar_kin.value < 300")
        parsed_filter = ['st_vel', '<', '300']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_st_vel_ivar_eq_0(self):
        desired = ("stellar_kin_parameter.name == 'VEL' and stellar_kin.ivar = 0")
        parsed_filter = ['st_vel_ivar', '=', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_st_velocity_ivar_eq_0(self):
        desired = ("stellar_kin_parameter.name == 'VEL' and stellar_kin.ivar = 0")
        parsed_filter = ['st_velocity_ivar', '=', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_st_sig_lt_200(self):
        desired = ("stellar_kin_parameter.name == 'SIGMA' and stellar_kin.value < 200")
        parsed_filter = ['st_sig', '<', '200']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_st_sig_ivar_eq_0(self):
        desired = ("stellar_kin_parameter.name == 'SIGMA' and stellar_kin.ivar = 0")
        parsed_filter = ['st_sig_ivar', '=', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_st_sigma_ivar_eq_0(self):
        desired = ("stellar_kin_parameter.name == 'SIGMA' and stellar_kin.ivar = 0")
        parsed_filter = ['st_sigma_ivar', '=', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_halpha_vel_ivar_eq_0(self):
        desired = ("emline_type.name == 'Ha' and emline_type.rest_wavelength == 6564 and "
                   "emline_parameter.name == 'GVEL' and emline.ivar = 0")
        parsed_filter = ['Ha_vel_ivar', '=', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_halpha_vel_le_200(self):
        desired = ("emline_type.name == 'Ha' and emline_type.rest_wavelength == 6564 and "
                   "emline_parameter.name == 'GVEL' and emline.value <= 200")
        parsed_filter = ['Ha_vel', '<=', '200']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_halpha_flux_ge_2(self):
        desired = ("emline_type.name == 'Ha' and emline_type.rest_wavelength == 6564 and "
                   "emline_parameter.name == 'GFLUX' and emline.value >= 2")
        parsed_filter = ['Ha_flux', '>=', '2']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_halpha_gflux_ne_2(self):
        desired = ("emline_type.name == 'Ha' and emline_type.rest_wavelength == 6564 and "
                   "emline_parameter.name == 'GFLUX' and emline.value != 2")
        parsed_filter = ['Ha_GFLUX', '!=', '2']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_oiii5008_flux_ivar_ne_0(self):
        desired = ("emline_type.name == 'OIII' and emline_type.rest_wavelength == 5008 and "
                   "emline_parameter.name == 'GFLUX' and emline.ivar != 0")
        parsed_filter = ['oiii5008_flux_ivar', '!=', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_oiii5008_flux_mask_ne_0(self):
        desired = ("emline_type.name == 'OIII' and emline_type.rest_wavelength == 5008 and "
                   "emline_parameter.name == 'GFLUX' and emline.mask != 0")
        parsed_filter = ['OIII5008_flux_mask', '!=', '0']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)

    def test_hbeta_flux_gt_1(self):
        desired = ("emline_type.name == 'Hb' and emline_type.rest_wavelength == 4862 and "
                   "emline_parameter.name == 'GFLUX' and emline.value > 1")
        parsed_filter = ['Hb_flux', '>', '1']
        actual = query_utils.expand(*parsed_filter)
        self.assertEqual(actual, desired)


if __name__ == '__main__':
    unittest.main()
