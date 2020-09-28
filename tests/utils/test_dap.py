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

# from marvin.utils.datamodel.dap import Property, MapsPropertyList, get_dap_datamodel
# from tests import UseReleases
# from marvin import config

# @UseReleases('MPL-5')
# class TestMapsProperties(object):
#
#     def test_dap_datamodel_mpl4(self, release):
#         datamodel = get_dap_datamodel('1.1.1')
#         assert len(datamodel) == 10
#         assert datamodel.version == '1.1.1'
#         assert isinstance(datamodel, MapsPropertyList)
#         assert isinstance(datamodel[0], Property)
#
#     def test_MapsPropertyList(self, release):
#         drpver, dapver = config.lookUpVersions(release)
#         datamodel = get_dap_datamodel(dapver)
#         assert datamodel.version == dapver
#         assert 'EMLINE_GFLUX' in datamodel
#         assert not ('emline_bad' in datamodel)
#         assert isinstance(datamodel['emline_gflux'], Property)
#         assert isinstance(datamodel == 'emline_gflux', Property)
#         assert (datamodel == 'emline_bad') is None, Property
#
#     def test_MapsPropertyList_get(self, release):
#         drpver, dapver = config.lookUpVersions(release)
#         datamodel = get_dap_datamodel(dapver)
#         assert datamodel.get('badname_badchannel') is None
#         assert datamodel.get('emline_gflux') is None
#         assert datamodel.get('emline_gflux_badchannel') is None
#
#         maps_prop, channel = datamodel.get('emline_gflux_ha_6564')
#         assert isinstance(maps_prop, Property)
#         assert maps_prop.name == 'emline_gflux'
#         assert channel == 'ha_6564'
