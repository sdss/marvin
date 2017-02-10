#!/usr/bin/env python3
# encoding: utf-8
#
# test_dap.py
#
# Created by José Sánchez-Gallego on 19 Sep 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import unittest

from marvin import config
from marvin.utils.dap.datamodel import MapsProperty, MapsPropertyList, get_dap_datamodel


class TestMapsProperties(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config.setMPL('MPL-5')
        config.use_sentry = False
        config.add_github_message = False

    def test_dap_datamodel_mpl4(self):
        datamodel = get_dap_datamodel('1.1.1')
        self.assertEqual(len(datamodel), 10)
        self.assertEqual(datamodel.version, '1.1.1')
        self.assertIsInstance(datamodel, MapsPropertyList)
        self.assertIsInstance(datamodel[0], MapsProperty)

    def test_MapsPropertyList(self):
        datamodel = get_dap_datamodel()
        self.assertEqual(datamodel.version, '2.0.2')
        self.assertTrue('EMLINE_GFLUX' in datamodel)
        self.assertFalse('emline_bad' in datamodel)
        self.assertIsInstance(datamodel['emline_gflux'], MapsProperty)
        self.assertIsInstance(datamodel == 'emline_gflux', MapsProperty)
        self.assertIsNone(datamodel == 'emline_bad', MapsProperty)

    def test_MapsPropertyList_get(self):
        datamodel = get_dap_datamodel()
        self.assertIsNone(datamodel.get('badname_badchannel'))
        self.assertIsNone(datamodel.get('emline_gflux'))
        self.assertIsNone(datamodel.get('emline_gflux_badchannel'))

        maps_prop, channel = datamodel.get('emline_gflux_oii_3727')
        self.assertIsInstance(maps_prop, MapsProperty)
        self.assertEqual(maps_prop.name, 'emline_gflux')
        self.assertEqual(channel, 'oii_3727')


if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)