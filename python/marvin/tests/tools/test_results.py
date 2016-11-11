#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import unittest
from marvin import config
from marvin.tools.query import Query, Results
from marvin.api.api import Interaction
from collections import OrderedDict, namedtuple
from marvin.tools.cube import Cube


class TestResults(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        config.switchSasUrl('local')

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
        config.sasurl = self.init_sasurl
        config.mode = self.init_mode
        config.urlmap = self.init_urlmap
        config.setMPL('MPL-4')

        self.filter = 'nsa.z < 0.02 and ifu.name=19*'
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
        self.res = (u'1-22438', 7992, u'7992-1901', u'1901', 0.016383046284318)
        self.q = Query(searchfilter=self.filter)

    def tearDown(self):
        pass

    def _setRemote(self, mode='local'):
        config.switchSasUrl(mode)
        response = Interaction('api/general/getroutemap', request_type='get')
        config.urlmap = response.getRouteMap()

        self.q = Query(searchfilter=self.filter, mode='remote')

    def _check_cols(self, mode='local'):
        r = self.q.run()
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
        r = self.q.run()
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
        r = self.q.run()
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
        r = self.q.run()
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
        r = self.q.run()
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
                self.assertEqual([name], output[0].keys())
            else:
                self.assertEqual(set(self.columns), set(output[0].keys()))
        elif ftype == 'dictlist':
            self.assertEqual(dict, type(output))
            self.assertEqual(list, type(output.get('cube.mangaid')))
            if name is not None:
                self.assertEqual([name], output.keys())
            else:
                self.assertEqual(set(self.columns), set(output.keys()))

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

    def _convertTool(self):
        r = self.q.run()
        r.convertToTool('cube')
        self.assertIsNotNone(r.objects)
        self.assertEqual(list, type(r.objects))
        self.assertEqual(True, isinstance(r.objects[0], Cube))

    def test_convert_to_tool_local(self):
        self._convertTool()

    def test_convert_to_tool_remote(self):
        self._setRemote()
        self._convertTool()

if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
