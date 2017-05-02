import numpy as np

import pytest

import marvin.utils.plot.map as mapplot

values = np.array([[1, -2, 3],
                   [-4, 5, 6],
                   [7, 8, 9]])

ivar = np.array([[4, 10, 1],
                 [0.01, 0.04, 0.0001],
                 [100000, 0, 1]])

mask_simple = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])

mask_binary = np.array([[0b000, 0b001, 0b010],
                        [0b011, 0b100, 0b101],
                        [0b110, 0b111, 0b000]])

mask_daplike = np.array([[0b0000000000000000000000000000000,
                          0b0000000000000000000000000000001,
                          0b0000000000000000000000000100000],
                         [0b0000000000000000000000001000001,
                          0b0000000000000000000000010000000,
                          0b1000000000000000000000000000001],
                         [0b0000000000000000000000000000000,
                          0b1000000000000000000000011100001,
                          0b0000000000000000000000000000000]])

nocov_mask = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 0, 1]])
nocov = np.ma.array(np.ones(values.shape), mask=nocov_mask)

bad_data_mask = np.array([[0, 0, 1],
                          [1, 1, 1],
                          [0, 1, 0]])

# SNR = value * np.sqrt(ivar)
# [[  2.00e+00,  -6.32e+00,   3.00e+00],
#  [ -4.00e-01,   1.00e+00,   6.00e-02],
#  [  2.21e+03,   0.00e+00,   9.00e+00]]

low_snr_mask_1_false = np.array([[0, 0, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])

low_snr_mask_3_false = np.array([[1, 0, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])

low_snr_mask_1_true = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]])

low_snr_mask_3_true = np.array([[1, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]])


class TestMasks(object):

    @pytest.mark.parametrize("values, mask, expected", [(values, mask_simple, nocov),
                                                        (values, mask_binary, nocov),
                                                        (values, mask_daplike, nocov)])
    def test_no_coverage_mask(self, values, mask, expected):
        nocov = mapplot.no_coverage_mask(values, mask)
        assert np.all(nocov == expected)

    @pytest.mark.parametrize("values, mask, expected", [(values, mask_daplike, bad_data_mask)])
    def test_bad_data_mask(self, values, mask, expected):
        bad_data = mapplot.bad_data_mask(values, mask)
        assert np.all(bad_data == expected)


    @pytest.mark.parametrize("values, ivar, snr_min, log_cb, expected",
                             [(values, ivar, 1, False, low_snr_mask_1_false),
                              (values, ivar, 3, False, low_snr_mask_3_false),
                              (values, ivar, 1, True, low_snr_mask_1_true),
                              (values, ivar, 3, True, low_snr_mask_3_true)])
    def test_low_snr_mask(self, values, ivar, snr_min, log_cb, expected):
        low_snr = mapplot.low_snr_mask(values, ivar, snr_min, log_cb)
        assert np.all(low_snr == expected)





