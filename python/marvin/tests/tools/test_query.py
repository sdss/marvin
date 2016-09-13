#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import unittest
import copy
from marvin import config, marvindb
from marvin.tests import MarvinTest, skipIfNoDB
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
        p = 'cube.plate==8485 and junk.emline_gflux_ha_6564>25'
        q = Query(searchfilter=p)
        r = q.run()
        self.assertEqual(self.plate, r.results[0].__getattribute__('plate'))
        self.assertEqual(self.mangaid, r.results[0].__getattribute__('mangaid'))
        self.assertEqual(26.3447, r.results[0].__getattribute__('emline_gflux_ha_6564'))
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

    def test_dap_query_1(self):
        p = 'junk.emline_gflux_ha_6564 > 25'
        q = Query(searchfilter=p, returnparams=['junk.emline_gflux_hb_4862'])
        r = q.run()
        self.assertEqual(231, r.totalcount)

    def test_dap_query_2(self):
        p = 'npergood(junk.emline_gflux_ha_6564 > 5) >= 20'
        q = Query(searchfilter=p)
        r = q.run()
        self.assertEqual(10008, r.totalcount)
        tmp = r.getAll()
        mangaids = r.getListOf('mangaid')
        self.assertEqual(8, len(set(mangaids)))

if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
