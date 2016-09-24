#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import unittest
import copy
from marvin import config, marvindb
from marvin.core import MarvinError
from marvin.tests import MarvinTest, skipIfNoDB
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.spaxel import Spaxel
from marvin.tools.query import Query, doQuery
from marvin.api.api import Interaction


class TestQuery(MarvinTest):

    @classmethod
    def setUpClass(cls):
        cls.mangaid = '1-209232'
        cls.plate = 8485
        cls.plateifu = '8485-1901'
        cls.cubepk = 10179
        cls.ra = 232.544703894
        cls.dec = 48.6902009334

        cls.initconfig = copy.deepcopy(config)
        cls.init_mode = config.mode
        cls.init_sasurl = config.sasurl
        cls.init_urlmap = config.urlmap

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        cvars = ['drpver', 'dapver']
        for var in cvars:
            config.__setattr__(var, self.initconfig.__getattribute__(var))
        config.sasurl = self.init_sasurl
        config.mode = self.init_mode
        config.urlmap = self.init_urlmap
        config.setMPL('MPL-4')

    def tearDown(self):
        pass

    def test_Query_emptyinit(self):
        q = Query()
        self.assertIsNone(q.query)

    def test_Query_drpver_and_dapver(self):
        p = 'cube.plate==8485 and spaxelprop.emline_gflux_ha_6564>25'
        q = Query(searchfilter=p)
        r = q.run()
        tmp = r.sort('emline_gflux_ha_6564')
        self.assertEqual(self.plate, r.results[0].__getattribute__('plate'))
        self.assertEqual(self.mangaid, r.results[0].__getattribute__('mangaid'))
        self.assertEqual(26.2344, r.results[0].__getattribute__('emline_gflux_ha_6564'))
        self.assertGreaterEqual(r.count, 6)
        self.assertIn('drpalias', str(q.query.whereclause))
        self.assertIn('dapalias', str(q.query.whereclause))

    def test_Query_only_drpver(self):
        p = 'cube.plate==8485 and spaxel.x > 5'
        q = Query(searchfilter=p)
        r = q.run()
        self.assertEqual(self.plate, r.results[0].__getattribute__('plate'))
        self.assertEqual(self.mangaid, r.results[0].__getattribute__('mangaid'))
        self.assertIn('drpalias', str(q.query.whereclause))
        self.assertNotIn('dapalias', str(q.query.whereclause))

    def test_Query_drp_but_nomaps(self):
        p = 'cube.plate < 8400'
        q = Query(searchfilter=p)
        r = q.run()
        self.assertIn('drpalias', str(q.query.whereclause))
        self.assertNotIn('dapalias', str(q.query.whereclause))

    def _queryparams(self, p, params, queryparams):
        q = Query(searchfilter=p)
        self.assertListEqual(params, q.params)
        keys = [s.key for s in q.queryparams]
        self.assertListEqual(queryparams, keys)

    def test_Query_queryparams_onlyfilter(self):
        p = 'nsa.z < 0.12 and ifu.name = 19*'
        params = ['cube.mangaid', 'cube.plate', 'ifu.name', 'nsa.z']
        qps = ['mangaid', 'plate', 'name', 'z']
        self._queryparams(p, params, qps)

    def _setRemote(self):
        config.sasurl = 'http://localhost:5000/marvin2/'
        response = Interaction('api/general/getroutemap', request_type='get')
        config.urlmap = response.getRouteMap()

    def test_Query_remote_mpl4(self):
        self._setRemote()
        p = 'nsa.z < 0.12 and ifu.name = 19*'
        q = Query(searchfilter=p, mode='remote')
        r = q.run()
        self.assertEqual([], q.joins)
        self.assertEqual(151, r.totalcount)  # MPL-4 count

    def test_Query_remote_mpl5(self):
        config.setMPL('MPL-5')
        self._setRemote()
        p = 'nsa.z < 0.12 and ifu.name = 19*'
        q = Query(searchfilter=p, mode='remote')
        r = q.run()
        self.assertEqual([], q.joins)
        self.assertEqual(2, r.totalcount)  # MPL-4 count

    def _dap_query_1(self, count, table=None, name='emline_gflux_ha_6564', classname='SpaxelProp', allspax=None):
        classname = 'Clean{0}'.format(classname) if not allspax else classname
        if table:
            key = '{0}.{1}'.format(table, name)
        else:
            key = '{0}'.format(name)

        p = '{0} > 25'.format(key)
        q = Query(searchfilter=p, returnparams=['spaxelprop.emline_gflux_hb_4862'], allspaxels=allspax)
        r = q.run()
        self.assertEqual(classname, q.marvinform._param_form_lookup[key].Meta.model.__name__)
        self.assertEqual(count, r.totalcount)

    def _bad_query_1(self, errmsg, *args, **kwargs):
        with self.assertRaises(MarvinError) as cm:
            self._dap_query_1(*args, **kwargs)
        self.assertIn(errmsg, str(cm.exception))

    def test_dap_query_1_normal(self):
        self._dap_query_1(231, table='spaxelprop', allspax=True)

    def test_dap_query_1_haflux(self):
        self._dap_query_1(231, name='haflux', allspax=True)

    def test_dap_query_1_normal_clean(self):
        self._dap_query_1(231, table='spaxelprop')

    def test_dap_query_1_haflux_clean(self):
        self._dap_query_1(231, name='haflux')

    def test_dap_query_1_badshortcut(self):
        errmsg = "Table 'spaxelprop' does not have a field named 'emline_gflux_ha_6564'"
        self._bad_query_1(errmsg, 231, table='spaxelprop5', allspax=True)

    def test_dap_query_1_wrongname(self):
        errmsg = 'spaxelprop5.emline_gluxf_ha does not match any column'
        self._bad_query_1(errmsg, 231, table='spaxelprop5', name='emline_gluxf_ha', allspax=True)

    def test_dap_query_1_wrongtable(self):
        errmsg = 'prop5.emline_gflux_ha_6564 does not match any column'
        self._bad_query_1(errmsg, 231, table='prop5', allspax=True)

    def test_dap_query_1_normal_mpl5(self):
        config.setMPL('MPL-5')
        self._dap_query_1(18, table='spaxelprop', classname='SpaxelProp5', allspax=True)

    def test_dap_query_1_haflux_mpl5(self):
        config.setMPL('MPL-5')
        self._dap_query_1(18, name='haflux', classname='SpaxelProp5', allspax=True)

    def test_dap_query_1_normal_mpl5_clean(self):
        config.setMPL('MPL-5')
        self._dap_query_1(18, table='spaxelprop', classname='SpaxelProp5')

    def test_dap_query_1_haflux_mpl5_clean(self):
        config.setMPL('MPL-5')
        self._dap_query_1(18, name='haflux', classname='SpaxelProp5')

    def test_dap_query_1_sp5_mpl5(self):
        config.setMPL('MPL-5')
        self._dap_query_1(18, table='spaxelprop5', classname='SpaxelProp5', allspax=True)

    def test_dap_query_2(self):
        p = 'npergood(spaxelprop.emline_gflux_ha_6564 > 5) >= 20'
        q = Query(searchfilter=p)
        r = q.run()
        self.assertEqual(8, r.totalcount)

    def test_dap_query_2_remote(self):
        self._setRemote()
        p = 'npergood(spaxelprop.emline_gflux_ha_6564 > 5) >= 20'
        q = Query(searchfilter=p)
        r = q.run()
        self.assertEqual(8, r.totalcount)

    def _query_return_type(self, rt=None):

        if rt == 'cube':
            tool = Cube

        config.setMPL('MPL-5')
        p = 'haflux > 25'
        q = Query(searchfilter=p, returntype=rt)
        r = q.run()
        self.assertIsNotNone(r.objects)
        self.assertEqual(len(r.results), len(r.objects))
        self.assertEqual(True, isinstance(r.objects[0], tool))

    def test_query_returntype_cube(self):
        self._query_return_type(rt='cube')

if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
