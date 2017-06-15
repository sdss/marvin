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
from marvin.tests import MarvinTest
from marvin.utils.dap.datamodel import MapsProperty, MapsPropertyList, get_dap_datamodel


class TestMapsProperties(MarvinTest):

    @classmethod
    def setUpClass(cls):
        super(TestMapsProperties, cls).setUpClass()
        config.setMPL('MPL-5')

    def test_dap_datamodel_mpl4(self):
        datamodel = get_dap_datamodel('1.1.1')
        assert len(datamodel) == 10
        assert datamodel.version == '1.1.1'
        assert isinstance(datamodel, MapsPropertyList)
        assert isinstance(datamodel[0], MapsProperty)

    def test_MapsPropertyList(self):
        datamodel = get_dap_datamodel()
        assert datamodel.version == '2.0.2'
        assert 'EMLINE_GFLUX' in datamodel
        assert not ('emline_bad' in datamodel)
        assert isinstance(datamodel['emline_gflux'], MapsProperty)
        assert isinstance(datamodel == 'emline_gflux', MapsProperty)
        assert (datamodel == 'emline_bad') is None, MapsProperty

    def test_MapsPropertyList_get(self):
        datamodel = get_dap_datamodel()
        assert datamodel.get('badname_badchannel') is None
        assert datamodel.get('emline_gflux') is None
        assert datamodel.get('emline_gflux_badchannel') is None

        maps_prop, channel = datamodel.get('emline_gflux_oii_3727')
        assert isinstance(maps_prop, MapsProperty)
        assert maps_prop.name == 'emline_gflux'
        assert channel == 'oii_3727'


if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
