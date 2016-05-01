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

    def tearDown(self):
        pass

    def test_Query_emptyinit(self):
        q = Query()
        self.assertIsNone(q.query)

    def test_Query_drpver_and_dapver(self):
        p = 'cube.plate==8485 and emline_type.name==Ha'
        q = Query(searchfilter=p)
        r = q.run()
        self.assertEqual(self.plate, r.results[0].__getattribute__('plate'))
        self.assertEqual(self.mangaid, r.results[0].__getattribute__('mangaid'))
        self.assertEqual('Ha', r.results[0].__getattribute__('name'))
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
        p = 'nsa_redshift < 0.012 and ifu.name = 19*'
        params = ['cube.mangaid', 'ifu.name', 'nsa_redshift']
        qps = ['mangaid', 'name', 'nsa_redshift']
        self._queryparams(p, params, qps)

    def _setRemote(self):
        config.sasurl = 'http://cd057661.ngrok.io'
        response = Interaction('api/general/getroutemap', request_type='get')
        config.urlmap = response.getRouteMap()

    def test_Query_remote(self):
        self._setRemote()
        p = 'nsa_redshift < 0.012 and ifu.name = 19*'
        q = Query(searchfilter=p, mode='remote')
        r = q.run()
        self.assertEqual([], q.joins)
        self.assertEqual(64, r.count)


if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
