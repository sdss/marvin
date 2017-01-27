#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import unittest
import os
from marvin import config
from marvin.tools.query import Query, Results
from marvin.core.exceptions import MarvinError
from marvin.api.api import Interaction
from collections import OrderedDict, namedtuple
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube
from marvin.tests import MarvinTest, skipIfNoBrian


class TestResultsBase(MarvinTest):

    @classmethod
    def setUpClass(cls):

        super(TestResultsBase, cls).setUpClass()
        cls.mangaid = '1-209232'
        cls.plate = 8485
        cls.plateifu = '8485-1901'
        cls.cubepk = 10179
        cls.ra = 232.544703894
        cls.dec = 48.6902009334

        cls.init_mode = config.mode
        cls.init_sasurl = config.sasurl
        cls.init_urlmap = config.urlmap

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        config.switchSasUrl('local')
        config.sasurl = self.init_sasurl
        self.mode = self.init_mode
        config.urlmap = self.init_urlmap
        config.setMPL('MPL-5')
        config.forceDbOn()

        self.filter = 'nsa.z < 0.1 and cube.plate==8485'
        self.columns = ['cube.mangaid', 'cube.plate', 'cube.plateifu', 'ifu.name', 'nsa.z']
        self.remotecols = [u'mangaid', u'plate', u'plateifu', u'name', u'z']
        self.coltoparam = OrderedDict([('mangaid', 'cube.mangaid'),
                                       ('plate', 'cube.plate'),
                                       ('plateifu', 'cube.plateifu'),
                                       ('name', 'ifu.name'),
                                       ('z', 'nsa.z')])
        self.paramtocol = OrderedDict([('cube.mangaid', 'mangaid'),
                                       ('cube.plate', 'plate'),
                                       ('cube.plateifu', 'plateifu'),
                                       ('ifu.name', 'name'),
                                       ('nsa.z', 'z')])

        self.res = (u'1-209232', 8485, u'8485-1901', u'1901', 0.0407447)

        self.resdict = {'1': (u'1-43148', 8135, u'8135-6101', u'6101', 0.0108501),
                        '10': (u'1-167079', 8459, u'8459-1901', u'1901', 0.015711),
                        '11': (u'1-167075', 8459, u'8459-12704', u'12704', 0.0158584),
                        '21': (u'1-113567', 7815, u'7815-12701', u'12701', 0.0167432),
                        '31': (u'1-322048', 8552, u'8552-12705', u'12705', 0.0172298),
                        '36': (u'1-252151', 8335, u'8335-9102', u'9102', 0.0174864),
                        '41': (u'1-378182', 8134, u'8134-12705', u'12705', 0.0178659),
                        '46': (u'1-252126', 8335, u'8335-3703', u'3703', 0.0181555)}

        self.q = Query(searchfilter=self.filter, mode=self.mode)

    def tearDown(self):
        pass

    def _setRemote(self, mode='local', limit=100):
        config.switchSasUrl(mode)
        self.mode = 'remote'
        response = Interaction('api/general/getroutemap', request_type='get')
        config.urlmap = response.getRouteMap()

        self.q = Query(searchfilter=self.filter, mode='remote', limit=limit)

    def _run_query(self):
        r = self.q.run()
        r.sort('z', order='desc')
        plateifu = r.getListOf('plateifu')
        index = plateifu.index(self.plateifu)
        newres = r.results[index]
        self.assertEqual(self.res, newres)
        return r


class TestResults(TestResultsBase):

    def _check_cols(self, mode='local'):
        r = self._run_query()
        if mode == 'local':
            self.assertEqual(r.columns, self.columns)
        elif mode == 'remote':
            self.assertEqual(r.columns, self.remotecols)
        self.assertEqual(r.coltoparam, self.coltoparam)
        self.assertEqual(r.paramtocol, self.paramtocol)

    def test_columns_local(self):
        self._check_cols()

    def test_columns_remote(self):
        self._setRemote()
        self._check_cols(mode='remote')

    def _getattribute(self, mode='local'):
        r = self._run_query()
        res = r.results[0]

        cols = self.columns if mode == 'local' else self.remotecols
        for i, name in enumerate(cols):
            self.assertEqual(self.res[i], res.__getattribute__(name))

    def test_res_getattribute_local(self):
        self._getattribute()

    def test_res_getattribute_remote(self):
        self._setRemote()
        self._getattribute(mode='remote')

    def _refname(self, mode='local'):
        r = self._run_query()
        cols = self.columns if mode == 'local' else self.remotecols
        for i, expname in enumerate(cols):
            self.assertEqual(expname, r._getRefName(self.columns[i]))
            self.assertEqual(expname, r._getRefName(self.remotecols[i]))

    def test_refname_local(self):
        self._refname()

    def test_refname_remote(self):
        self._setRemote()
        self._refname(mode='remote')

    def _get_list(self, name):
        r = self._run_query()
        obj = r.getListOf(name)
        self.assertIsNotNone(obj)
        self.assertEqual(list, type(obj))

    def test_getList_local(self):
        for i, col in enumerate(self.columns):
            self._get_list(col)
            self._get_list(self.remotecols[i])

    def test_getList_remote(self):
        self._setRemote()
        for i, col in enumerate(self.columns):
            self._get_list(col)
            self._get_list(self.remotecols[i])

    def _get_dict(self, name=None, ftype='listdict'):
        r = self._run_query()
        # get output
        if name is not None:
            output = r.getDictOf(name, format_type=ftype)
        else:
            output = r.getDictOf(format_type=ftype)

        # test output
        if ftype == 'listdict':
            self.assertEqual(list, type(output))
            self.assertEqual(dict, type(output[0]))
            if name is not None:
                self.assertEqual([name], list(output[0]))
            else:
                self.assertEqual(set(self.columns), set(output[0]))
        elif ftype == 'dictlist':
            self.assertEqual(dict, type(output))
            self.assertEqual(list, type(output.get('cube.mangaid')))
            if name is not None:
                self.assertEqual([name], list(output.keys()))
            else:
                self.assertEqual(set(self.columns), set(output))

    def test_get_dict_local(self):
        self._get_dict()

    def test_get_dict_local_param(self):
        self._get_dict(name='cube.mangaid')

    def test_get_dict_local_dictlist(self):
        self._get_dict(ftype='dictlist')

    def test_get_dict_local_dictlist_param(self):
        self._get_dict(name='cube.mangaid', ftype='dictlist')

    def test_get_dict_remote(self):
        self._setRemote()
        self._get_dict()

    def test_get_dict_remote_param(self):
        self._setRemote()
        self._get_dict(name='cube.mangaid')

    def test_get_dict_remote_dictlist(self):
        self._setRemote()
        self._get_dict(ftype='dictlist')

    def test_get_dict_remote_dictlist_param(self):
        self._setRemote()
        self._get_dict(name='cube.mangaid', ftype='dictlist')


class TestResultsConvertTool(TestResultsBase):

    def _convertTool(self, tooltype='cube'):

        if tooltype == 'cube':
            marvintool = Cube
        elif tooltype == 'maps':
            marvintool = Maps
        elif tooltype == 'modelcube':
            marvintool = ModelCube
        else:
            marvintool = Cube

        r = self._run_query()
        r.convertToTool(tooltype, limit=1)
        self.assertIsNotNone(r.objects)
        self.assertEqual(list, type(r.objects))
        self.assertEqual(True, isinstance(r.objects[0], marvintool))

    def test_convert_to_tool_local(self):
        self._convertTool()

    def test_convert_tool_no_spaxel(self):
        with self.assertRaises(AssertionError) as cm:
            self._convertTool('spaxel')
        errmsg = 'Parameters must include spaxelprop.x and y in order to convert to Marvin Spaxel'
        self.assertIn(errmsg, str(cm.exception))

    def test_convert_tool_map(self):
        self._convertTool('maps')

    def test_convert_tool_modelcube(self):
        self._convertTool('modelcube')

    def test_convert_tool_no_modelcube(self):
        config.setRelease('MPL-4')
        with self.assertRaises(MarvinError) as cm:
            self._convertTool('modelcube')
        errmsg = "ModelCube requires at least dapver='2.0.2'"
        self.assertIn(errmsg, str(cm.exception))

    def test_convert_to_tool_remote(self):
        self._setRemote()
        self._convertTool()

    def test_convert_tool_auto(self):
        self._setRemote()
        r = self._run_query()
        r.convertToTool('cube', mode='auto')
        self.assertEqual('remote', r.mode)
        self.assertEqual('local', r.objects[0].mode)
        self.assertEqual('db', r.objects[0].data_origin)

    def test_convert_tool_auto_nodb(self):
        self._setRemote()
        config.forceDbOff()
        r = self._run_query()
        r.convertToTool('cube', mode='auto', limit=1)
        self.assertEqual('remote', r.mode)
        self.assertEqual('local', r.objects[0].mode)
        self.assertEqual('file', r.objects[0].data_origin)


class TestResultsPickling(TestResultsBase):

    @classmethod
    def setUpClass(cls):

        for fn in ['~/test_results.mpf']:
            if os.path.exists(fn):
                os.remove(fn)

        super(TestResultsPickling, cls).setUpClass()

    def setUp(self):
        self._files_created = []
        super(TestResultsPickling, self).setUp()

    def tearDown(self):

        for fp in self._files_created:
            if os.path.exists(fp):
                os.remove(fp)

    def test_pickle_results(self):
        self._setRemote()
        r = self._run_query()
        path = r.save('results_test.mpf', overwrite=True)
        self._files_created.append(path)
        self.assertTrue(os.path.exists(path))

        r = None
        self.assertIsNone(r)

        r = Results.restore(path)
        self.assertEqual('nsa.z < 0.1 and cube.plate==8485', r.searchfilter)
        self.assertEqual('remote', r.mode)


class TestResultsPage(TestResultsBase):

    def _setrun_query(self, limit=10):
        config.setRelease("MPL-4")
        self.filter = 'nsa.z < 0.1'
        self._setRemote(limit=limit)
        r = self.q.run()
        r.sort('z')
        self.assertEqual(1213, r.totalcount)
        self.assertEqual(limit, r.count)
        self.assertEqual(limit, len(r.results))
        self.assertEqual(limit, r.limit)
        self.assertEqual(limit, r.chunk)
        return r

    @skipIfNoBrian
    def test_sort_res(self):
        r = self._setrun_query(limit=10)
        self.assertEqual(r.results[0], self.resdict['1'])
        self.assertEqual(r.results[9], self.resdict['10'])

    def _get_set(self, r, go, chunk=None, index=None):
        if go == 'next':
            r.getNext() if not chunk else r.getNext(chunk=chunk)
        elif go == 'prev':
            r.getPrevious() if not chunk else r.getPrevious(chunk=chunk)
        elif go == 'set':
            r.getSubset(index) if not chunk else r.getSubset(index, limit=chunk)

        if chunk:
            self.assertEqual(chunk, r.chunk)
            self.assertEqual(chunk, len(r.results))

        return r

    @skipIfNoBrian
    def test_getNext_10(self):
        r = self._setrun_query(limit=10)
        r = self._get_set(r, 'next')
        self.assertEqual(r.results[0], self.resdict['11'])
        self.assertEqual(10, len(r.results))

    @skipIfNoBrian
    def test_getNext_20(self):
        r = self._setrun_query(limit=10)
        r = self._get_set(r, 'next', chunk=20)
        self.assertEqual(r.results[0], self.resdict['11'])
        self.assertEqual(r.results[10], self.resdict['21'])

    @skipIfNoBrian
    def test_getPrevious_10(self):
        r = self._setrun_query(limit=10)
        r = self._get_set(r, 'set', index=30)
        r = self._get_set(r, 'prev')
        self.assertEqual(r.results[0], self.resdict['21'])
        self.assertEqual(10, len(r.results))

    @skipIfNoBrian
    def test_getPrevious_20(self):
        r = self._setrun_query(limit=10)
        r = self._get_set(r, 'set', index=45)
        r = self._get_set(r, 'prev', chunk=20)
        self.assertEqual(r.results[5], self.resdict['31'])
        self.assertEqual(r.results[10], self.resdict['36'])
        self.assertEqual(r.results[15], self.resdict['41'])
        self.assertEqual(20, len(r.results))

    @skipIfNoBrian
    def test_getSubset_10(self):
        r = self._setrun_query(limit=10)
        r = self._get_set(r, 'set', index=35)
        self.assertEqual(10, len(r.results))
        self.assertEqual(r.results[0], self.resdict['36'])

    @skipIfNoBrian
    def test_getSubset_20(self):
        r = self._setrun_query(limit=10)
        r = self._get_set(r, 'set', chunk=20, index=30)
        self.assertEqual(r.results[0], self.resdict['31'])
        self.assertEqual(r.results[15], self.resdict['46'])

if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
