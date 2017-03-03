#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import unittest
import os
from marvin import config, marvindb
from marvin import bconfig
from marvin.core.exceptions import MarvinError
from marvin.tests import MarvinTest, skipIfNoDB
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.spaxel import Spaxel
from marvin.tools.query import Query, doQuery
from marvin.api.api import Interaction


class TestQueryBase(MarvinTest):

    @classmethod
    def setUpClass(cls):
        super(TestQueryBase, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self._reset_the_config()
        self.set_sasurl('local', port=5000)
        self.mode = self.init_mode
        config.setMPL('MPL-4')
        config.forceDbOn()

    def tearDown(self):
        pass

    def _set_remote(self, mode='local'):
        self.mode = 'remote'


class TestQuery(TestQueryBase):

    def test_Query_emptyinit(self):
        q = Query()
        self.assertIsNone(q.query)

    def _query_versions(self, mode='local', release=None):

        release = config.release
        drpver, dapver = config.lookUpVersions(release=release)

        p = 'haflux > 25'

        if '4' in release:
            name = 'CleanSpaxelProp'
        else:
            name = 'CleanSpaxelProp{0}'.format(release.split('-')[1])

        if release:
            q = Query(searchfilter=p, mode=mode, release=release)
        else:
            q = Query(searchfilter=p, mode=mode)

        self.assertEqual(q._drpver, drpver)
        self.assertEqual(q._dapver, dapver)
        self.assertEqual(q._release, release)
        self.assertEqual(q.marvinform._release, release)
        self.assertEqual(q.marvinform._param_form_lookup._release, release)
        self.assertEqual(q.marvinform._param_form_lookup['spaxelprop.file'].Meta.model.__name__, name)

    def test_query_versions_local(self):
        self._query_versions(mode='local')

    def test_query_versions_remote(self):
        self._set_remote()
        self._query_versions(mode='remote')

    def test_query_versions_local_othermpl(self):
        vers = ('v2_0_1', '2.0.2', 'MPL-5')
        self._query_versions(mode='local', release='MPL-5')

    def test_query_versions_remote_othermpl(self):
        self._set_remote()
        vers = ('v2_0_1', '2.0.2', 'MPL-5')
        self._query_versions(mode='remote', release='MPL-5')

    def test_query_versions_remote_utah(self):
        self._set_remote(mode='utah')
        self._query_versions(mode='remote')

    def test_query_versions_local_mpl5(self):
        config.setMPL('MPL-5')
        self._query_versions(mode='local')

    def test_query_versions_remote_mpl5(self):
        config.setMPL('MPL-5')
        self._set_remote()
        self._query_versions(mode='remote')

    def test_query_versions_remote_utah_mpl5(self):
        config.setMPL('MPL-5')
        self._set_remote(mode='utah')
        self._query_versions(mode='remote')

    def test_Query_drpver_and_dapver(self):
        p = 'cube.plate==8485 and emline_gflux_ha_6564>25'
        q = Query(searchfilter=p)
        r = q.run()
        tmp = r.sort('emline_gflux_ha_6564')
        self.assertEqual(self.plate, r.getListOf('cube.plate')[0])
        self.assertEqual(self.mangaid, r.getListOf('cube.mangaid')[0])
        self.assertEqual(26.112, r.getListOf('emline_gflux_ha_6564')[0])
        self.assertGreaterEqual(r.count, 6)
        self.assertIn('drpalias', str(q.query.whereclause))
        self.assertIn('dapalias', str(q.query.whereclause))

    def test_Query_only_drpver(self):
        p = 'cube.plate==8485 and spaxel.x > 5'
        q = Query(searchfilter=p)
        r = q.run()
        self.assertEqual(self.plate, r.getListOf('cube.plate')[0])
        self.assertEqual(self.mangaid, r.getListOf('cube.mangaid')[0])
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
        params = ['cube.mangaid', 'cube.plate', 'cube.plateifu', 'ifu.name', 'nsa.z']
        qps = ['mangaid', 'plate', 'plateifu', 'name', 'z']
        self._queryparams(p, params, qps)

    def test_Query_remote_mpl4(self):
        self._set_remote()
        p = 'nsa.z < 0.12 and ifu.name = 19*'
        q = Query(searchfilter=p, mode='remote')
        r = q.run()
        self.assertEqual([], q.joins)
        self.assertEqual(151, r.totalcount)  # MPL-4 count

    def test_Query_remote_mpl5(self):
        config.setMPL('MPL-5')
        self._set_remote()
        p = 'nsa.z < 0.12 and ifu.name = 19*'
        q = Query(searchfilter=p, mode='remote')
        r = q.run()
        self.assertEqual([], q.joins)
        self.assertEqual(2, r.totalcount)  # MPL-4 count

    def _fail_query(self, errmsg, tracemsg, *args, **kwargs):
        with self.assertRaises(MarvinError) as cm:
            q = Query(**kwargs)
            r = q.run()
        self.assertIn(errmsg, str(cm.exception))
        self.assertIsNotNone(config._traceback)
        self.assertIsNotNone(bconfig.traceback)
        self.assertIn(tracemsg, config._traceback)

    def test_Query_remote_fail_traceback(self):
        self._set_remote(mode='local')
        p = 'nsa.z < 0.1'
        rt = 'nsa.hello'
        errmsg = 'nsa.hello does not match any column'
        tracemsg = 'Query failed with'
        self._fail_query(errmsg, tracemsg, searchfilter=p, returnparams=rt, mode=self.mode)

    def test_Query_remote_no_traceback(self):
        self._set_remote(mode='local')
        p = 'nsa.z < 0.1'
        q = Query(searchfilter=p)
        r = q.run()
        self.assertIsNone(config._traceback)

    def _dap_query_1(self, count, table=None, name='emline_gflux_ha_6564', classname='SpaxelProp', allspax=None):
        classname = 'Clean{0}'.format(classname) if not allspax else classname
        if table:
            key = '{0}.{1}'.format(table, name)
        else:
            key = '{0}'.format(name)

        p = '{0} > 25 and cube.plate == 8485'.format(key)
        q = Query(searchfilter=p, returnparams=['spaxelprop.emline_gflux_hb_4862'], allspaxels=allspax)
        r = q.run()
        self.assertEqual(classname, q.marvinform._param_form_lookup[key].Meta.model.__name__)
        self.assertEqual(count, r.totalcount)

    def _bad_query_1(self, errmsg, *args, **kwargs):
        with self.assertRaises(MarvinError) as cm:
            self._dap_query_1(*args, **kwargs)
        self.assertIn(errmsg, str(cm.exception))

    def test_dap_query_1_normal(self):
        self._dap_query_1(19, table='spaxelprop', allspax=True)

    def test_dap_query_1_haflux(self):
        self._dap_query_1(19, name='haflux', allspax=True)

    def test_dap_query_1_normal_clean(self):
        self._dap_query_1(19, table='spaxelprop')

    def test_dap_query_1_haflux_clean(self):
        self._dap_query_1(19, name='haflux')

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
        self._set_remote()
        p = 'npergood(spaxelprop.emline_gflux_ha_6564 > 5) >= 20'
        q = Query(searchfilter=p, mode=self.mode)
        r = q.run()
        self.assertEqual(8, r.totalcount)


class TestQueryReturnType(TestQueryBase):

    def _query_return_type(self, rt=None, mode='local'):

        if rt == 'cube':
            tool = Cube
        elif rt == 'maps':
            tool = Maps
        elif rt == 'spaxel':
            tool = Spaxel

        if mode == 'remote':
            self._set_remote()

        config.setMPL('MPL-5')
        p = 'haflux > 25'
        q = Query(searchfilter=p, returntype=rt, mode=mode)
        r = q.run()
        self.assertIsNotNone(r.objects)
        self.assertEqual(18, r.count)
        self.assertEqual(len(r.results), len(r.objects))
        self.assertEqual(True, isinstance(r.objects[0], tool))

    def test_query_returntype_cube(self):
        self._query_return_type(rt='cube')

    def test_results_returntype_cube_limit(self):
        config.setMPL('MPL-5')
        p = 'haflux > 25'
        q = Query(searchfilter=p)
        r = q.run()
        r.convertToTool('cube', limit=5)
        self.assertIsNotNone(r.objects)
        self.assertEqual(18, r.count)
        self.assertEqual(5, len(r.objects))
        self.assertEqual(True, isinstance(r.objects[0], Cube))

    def test_query_returntype_cube_remote(self):
        self._query_return_type(rt='cube', mode='remote')

    def test_query_returntype_maps(self):
        self._query_return_type(rt='maps')

    def test_query_returntype_maps_remote(self):
        self._query_return_type(rt='maps', mode='remote')

    def test_query_returntype_spaxel(self):
        self._query_return_type(rt='spaxel')

    def test_query_returntype_spaxel_remote(self):
        self._query_return_type(rt='spaxel', mode='remote')


class TestQueryPickling(TestQueryBase):

    @classmethod
    def setUpClass(cls):

        for fn in ['~/test_query.mpf']:
            if os.path.exists(fn):
                os.remove(fn)

        super(TestQueryPickling, cls).setUpClass()

    def setUp(self):
        self._files_created = []
        super(TestQueryPickling, self).setUp()

    def tearDown(self):

        for fp in self._files_created:
            if os.path.exists(fp):
                os.remove(fp)

    def _pickle_query(self, mode=None, name=None):
        p = 'nsa.z < 0.1'
        q = Query(searchfilter=p, mode=mode)
        path = q.save(name, overwrite=True)
        self._files_created.append(path)
        self.assertTrue(os.path.exists(path))

        q = None
        self.assertIsNone(q)

        q = Query.restore(path)
        self.assertEqual('nsa.z < 0.1', q.searchfilter)
        self.assertEqual('remote', q.mode)

    def test_pickle_save_local(self):
        errmsg = 'save not available in local mode'
        with self.assertRaises(MarvinError) as cm:
            self._pickle_query(mode='local', name='test_query.mpf')
        self.assertIn(errmsg, str(cm.exception))

    def test_pickle_save_remote(self):
        self._set_remote()
        self._pickle_query(mode='remote', name='test_query.mpf')

    def test_pickle_save_auto(self):
        self._set_remote()
        config.forceDbOff()
        self._pickle_query(mode='auto', name='test_query.mpf')


class TestQueryParams(TestQueryBase):

    def _get_params(self, pdisp, mode='local', expcount=None, inlist=None, outlist=None):
        if mode == 'remote':
            self._set_remote()
        q = Query(mode=mode)

        if pdisp == 'all':
            keys = q.get_available_params()
            self.assertGreaterEqual(len(keys), expcount)
        elif pdisp == 'best':
            keys = q.get_best_params()
            self.assertLessEqual(len(keys), expcount)

        for i in inlist:
            self.assertIn(i, keys)
        if outlist:
            for o in outlist:
                self.assertNotIn(o, keys)

    def test_get_available_params_local(self):
        expcount = 300
        inlist = ['nsa.tile', 'anime.anime', 'cube.ra', 'wcs.extname', 'spaxelprop.x', 'maskbit.bit']
        self._get_params('all', expcount=expcount, inlist=inlist)

    def test_get_available_params_remote(self):
        expcount = 300
        inlist = ['nsa.tile', 'anime.anime', 'cube.ra', 'wcs.extname', 'spaxelprop.x', 'maskbit.bit']
        self._get_params('all', mode='remote', expcount=expcount, inlist=inlist)

    def test_get_best_params_local(self):
        expcount = 300
        inlist = ['nsa.z', 'cube.ra', 'cube.dec', 'spaxelprop.emline_gflux_ha_6564']
        outlist = ['nsa.tile', 'anime.anime', 'wcs.extname', 'maskbit.bit']
        self._get_params('best', expcount=expcount, inlist=inlist, outlist=outlist)

    def test_get_best_params_remote(self):
        expcount = 300
        inlist = ['nsa.z', 'cube.ra', 'cube.dec', 'spaxelprop.emline_gflux_ha_6564']
        outlist = ['nsa.tile', 'anime.anime', 'wcs.extname', 'maskbit.bit']
        self._get_params('best', mode='remote', expcount=expcount, inlist=inlist, outlist=outlist)

    def test_read_best(self):
        q = Query(mode='local')
        bestkeys = q._read_best_params()
        self.assertEqual(type(bestkeys), list)
        exp = ['cube.ra', 'cube.dec', 'cube.plate', 'nsa.z', 'spaxelprop.x']
        for e in exp:
            self.assertIn(e, bestkeys)


class TestQueryModes(TestQueryBase):

    def _set_modes(self, expmode=None):
        p = 'nsa.z < 0.1 and cube.plate == 8485'
        q = Query(searchfilter=p, mode=self.mode)
        r = q.run()
        # this part selects 8485-1901
        r.sort('z')
        r.results = r.results[-4:]
        r.convertToTool('cube', limit=1)
        self.assertEqual(expmode, q.mode)
        self.assertEqual(expmode, r.mode)
        self.assertEqual(q.mode, r.mode)
        self.assertEqual(expmode, r.objects[0].mode)

    def test_mode_local(self):
        self.mode = 'local'
        self._set_modes(expmode='local')

    def test_mode_remote(self):
        self.mode = 'remote'
        self._set_modes(expmode='remote')

    def test_mode_auto(self):
        self.mode = 'auto'
        self._set_modes(expmode='local')

    def test_mode_local_nodb(self):
        config.forceDbOff()
        self.mode = 'local'
        errmsg = 'Query cannot be run in local mode'
        with self.assertRaises(MarvinError) as cm:
            self._set_modes(expmode='local')
        self.assertIn(errmsg, str(cm.exception))

    def test_mode_auto_nodb(self):
        config.forceDbOff()
        self.set_sasurl(loc='local')
        self.mode = 'auto'
        self._set_modes(expmode='remote')


if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
