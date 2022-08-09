#!/usr/bin/env python3
# encoding: utf-8
#
# test_dap.py
#
# Created by José Sánchez-Gallego on 19 Sep 2016.
# Adpated to pytest by Brett Andrews on 15 Jun 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
from marvin.utils.datamodel.dap import Property, PropertyList, datamodel
from marvin import config


class TestMapsProperties(object):

    def test_dap_datamodel_mpl4(self):
        dm = datamodel['1.1.1']
        assert len(dm.properties) == 92
        assert dm.release == '1.1.1'
        assert isinstance(dm.properties, PropertyList)
        assert isinstance(dm.properties[0], Property)

    def test_PropertyList(self, release):
        drpver, dapver = config.lookUpVersions(release)
        dm = datamodel[dapver]
        assert dm.release == dapver
        props = dm.properties
        assert 'EMLINE_GFLUX_HA_6564' in props
        assert isinstance(props['emline_gflux_ha'], Property)
        assert isinstance(props == 'emline_gflux_ha', Property)

        param = props['emline_gflux_ha_6564']
        assert isinstance(param, Property)
        assert param.name == 'emline_gflux'
        assert param.channel.name == 'ha_6564'

    def test_bad_property(self, release):
        drpver, dapver = config.lookUpVersions(release)
        dm = datamodel[dapver]
        with pytest.raises(ValueError, match=("cannot find a good match for 'emline_bad'. "
                                              "Your input value is too ambiguous.")):
            assert dm.properties['emline_bad']

