# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-12 18:18:54
# @Last modified by:   andrews
# @Last modified time: 2017-07-31 12:07:88

from __future__ import print_function, division, absolute_import
from marvin.tools.query import Query, Results
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.spaxel import Spaxel
from marvin.tools.modelcube import ModelCube
from marvin import config
from marvin.core.exceptions import MarvinError
from collections import OrderedDict, namedtuple
import pytest


myplateifu = '8485-1901'
cols = ['cube.mangaid', 'cube.plate', 'cube.plateifu', 'ifu.name', 'nsa.z']
remotecols = [u'mangaid', u'plate', u'plateifu', u'name', u'z']
coltoparam = OrderedDict([('mangaid', 'cube.mangaid'),
                          ('plate', 'cube.plate'),
                          ('plateifu', 'cube.plateifu'),
                          ('name', 'ifu.name'),
                          ('z', 'nsa.z')])
paramtocol = OrderedDict([('cube.mangaid', 'mangaid'),
                          ('cube.plate', 'plate'),
                          ('cube.plateifu', 'plateifu'),
                          ('ifu.name', 'name'),
                          ('nsa.z', 'z')])


# @pytest.fixture()
# def data(release):

#     rdata = {'MPL-5': {'resdict': {'1': (u'1-209151', 8485, u'8485-12702', u'12702', 0.0185246),
#                                    '2': (u'1-209191', 8485, u'8485-12701', u'12701', 0.0234253),
#                                    '3': (u'1-209113', 8485, u'8485-1902', u'1902', 0.0378877),
#                                    '4': (u'1-209232', 8485, u'8485-1901', u'1901', 0.0407447)},
#                        'row': (u'1-209232', 8485, u'8485-1901', u'1901', 0.0407447),
#                        'count': 4
#                        },
#              'MPL-4': {'resdict': {'1': (u'1-43148', 8135, u'8135-6101', u'6101', 0.0108501),
#                                    '10': (u'1-167079', 8459, u'8459-1901', u'1901', 0.015711),
#                                    '11': (u'1-167075', 8459, u'8459-12704', u'12704', 0.0158584),
#                                    '21': (u'1-113567', 7815, u'7815-12701', u'12701', 0.0167432),
#                                    '31': (u'1-322048', 8552, u'8552-12705', u'12705', 0.0172298),
#                                    '36': (u'1-252151', 8335, u'8335-9102', u'9102', 0.0174864),
#                                    '41': (u'1-378182', 8134, u'8134-12705', u'12705', 0.0178659),
#                                    '46': (u'1-252126', 8335, u'8335-3703', u'3703', 0.0181555)},
#                        'row': (u'1-209232', 8485, u'8485-1901', u'1901', 0.0407447),
#                        'count': 1213
#                        }
#              }

#     return rdata[release]


@pytest.fixture()
def limits(results):
    data = results.expdata['queries'][results.searchfilter]
    count = 10 if data['count'] >= 10 else data['count']
    return (10, count)


@pytest.fixture()
def results(query, request):
    searchfilter = request.param if hasattr(request, 'param') else 'nsa.z < 0.1 and cube.plate==8485'
    q = Query(searchfilter=searchfilter, mode=query.mode, limit=10, release=query._release)
    r = q.run()
    r.expdata = query.expdata
    yield r
    r = None


@pytest.fixture()
def columns(results):
    ''' returns the local or remote column syntax '''
    if results.mode == 'local':
        return cols
    elif results.mode == 'remote':
        return remotecols


class TestResultsColumns(object):

    def test_check_cols(self, results, columns):
        assert set(results.columns) == set(columns)
        assert results.coltoparam == coltoparam
        assert results.paramtocol == paramtocol

    @pytest.mark.parametrize('results', [('nsa.z < 0.1 and haflux > 25')], indirect=True)
    @pytest.mark.parametrize('colmns, pars', [(['spaxelprop.x', 'spaxelprop.y', 'emline_gflux_ha_6564', 'bintype.name', 'template.name'],
                                               ['x', 'y', 'emline_gflux_ha_6564', 'bintype_name', 'template_name'])])
    def test_check_withadded(self, results, colmns, pars, columns):
        newcols = cols + colmns
        assert set(results._params) == set(newcols)
        newcols = columns + colmns
        newrem = columns + pars
        if 'name' in newrem:
            newrem[newrem.index('name')] = 'ifu_name'
        assert set(results.columns) == set(newcols) or set(results.columns) == set(newrem)


class TestResultsGetParams(object):

    def test_get_attribute(self, results, columns):
        res = results.results[0]
        for i, name in enumerate(columns):
            assert res[i] == res.__getattribute__(name)

    def test_get_refname(self, results, columns):
        for i, name in enumerate(columns):
            assert name == results._getRefName(cols[i])
            assert name == results._getRefName(remotecols[i])

    @pytest.mark.parametrize('col',
                             [(c) for c in cols] + [(c) for c in remotecols],
                             ids=['cube.mangaid', 'cube.plate', 'cube.plateifu', 'ifu.name', 'nsa.z', u'mangaid', u'plate', u'plateifu', u'name', u'z'])
    def test_get_list(self, results, col):
        obj = results.getListOf(col)
        assert obj is not None
        assert isinstance(obj, list) is True

    @pytest.mark.parametrize('ftype', [('dictlist'), ('listdict')])
    @pytest.mark.parametrize('name', [(None), ('cube.mangaid'), ('nsa.z')], ids=['noname', 'mangaid', 'nsa.z'])
    def test_get_dict(self, results, ftype, name):
        output = results.getDictOf(name, format_type=ftype)

        if ftype == 'listdict':
            assert isinstance(output, list) is True
            assert isinstance(output[0], dict) is True
            if name is not None:
                assert set([name]) == set(list(output[0]))
            else:
                assert set(cols) == set(output[0])
        elif ftype == 'dictlist':
            assert isinstance(output, dict) is True
            assert isinstance(list(output.values())[0], list) is True
            if name is not None:
                assert set([name]) == set(list(output.keys()))
            else:
                assert set(cols) == set(output)


class TestResultsSort(object):

    @pytest.mark.parametrize('results', [('nsa.z < 0.1')], indirect=True)
    def test_sort(self, results, limits):
        if results.mode == 'local':
            pytest.skip('skipping now due to weird issue with local results not same as remote results')
        results.sort('z')
        limit, count = limits
        data = results.expdata['queries'][results.searchfilter]['sorted']
        assert tuple(data['1']) == results.results[0]
        assert tuple(data[str(count)]) == results.results[count - 1]


class TestResultsPaging(object):

    @pytest.mark.parametrize('results', [('nsa.z < 0.1')], indirect=True)
    def test_check_counts(self, results, limits):
        if results.mode == 'local':
            pytest.skip('skipping now due to weird issue with local results not same as remote results')
        results.sort('z')
        limit, count = limits
        data = results.expdata['queries'][results.searchfilter]
        assert results.totalcount == data['count']
        assert results.count == count
        assert len(results.results) == count
        assert results.limit == limit
        assert results.chunk == limit

    @pytest.mark.parametrize('results', [('nsa.z < 0.1')], indirect=True)
    @pytest.mark.parametrize('chunk, rows',
                             [(10, None),
                              (20, (10, 21))],
                             ids=['defaultchunk', 'chunk20'])
    def test_get_next(self, results, chunk, rows, limits):
        if results.mode == 'local':
            pytest.skip('skipping now due to weird issue with local results not same as remote results')
        limit, count = limits
        results.sort('z')
        results.getNext(chunk=chunk)
        data = results.expdata['queries'][results.searchfilter]['sorted']
        if results.count == results.totalcount:
            assert results.results[0] == tuple(data['1'])
            assert len(results.results) == count
        else:
            assert results.results[0] == tuple(data['11'])
            assert len(results.results) == chunk
            if rows:
                assert results.results[rows[0]] == tuple(data[str(rows[1])])

    @pytest.mark.parametrize('results', [('nsa.z < 0.1')], indirect=True)
    @pytest.mark.parametrize('index, chunk, rows',
                             [(30, 10, [(0, 21)]),
                              (45, 20, [(5, 31), (10, 36), (15, 41)])],
                             ids=['defaultchunk', 'chunk20'])
    def test_get_prev(self, results, index, chunk, rows, limits):
        if results.mode == 'local':
            pytest.skip('skipping now due to weird issue with local results not same as remote results')
        limit, count = limits
        results.sort('z')
        results.getSubset(index, limit=chunk)
        results.getPrevious(chunk=chunk)
        data = results.expdata['queries'][results.searchfilter]['sorted']
        if results.count == results.totalcount:
            assert results.results[0] == tuple(data['1'])
            assert len(results.results) == count
        elif results.count == 0:
            assert len(results.results) == 0
        else:
            assert len(results.results) == chunk
            if rows:
                for row in rows:
                    assert results.results[row[0]] == tuple(data[str(row[1])])

    @pytest.mark.parametrize('results', [('nsa.z < 0.1')], indirect=True)
    @pytest.mark.parametrize('index, chunk, rows',
                             [(35, 10, [(0, 36)]),
                              (30, 20, [(0, 31), (15, 46)])],
                             ids=['defaultchunk', 'chunk20'])
    def test_get_set(self, results, index, chunk, rows, limits):
        if results.mode == 'local':
            pytest.skip('skipping now due to weird issue with local results not same as remote results')
        limit, count = limits
        results.sort('z')
        results.getSubset(index, limit=chunk)
        data = results.expdata['queries'][results.searchfilter]['sorted']
        if results.count == results.totalcount:
            assert results.results[0] == tuple(data['1'])
            assert len(results.results) == count
        elif results.count == 0:
            assert len(results.results) == 0
        else:
            assert len(results.results) == chunk
            if rows:
                for row in rows:
                    assert results.results[row[0]] == tuple(data[str(row[1])])


class TestResultsPickling(object):

    def test_pickle_save(self, results, temp_scratch):
        file = temp_scratch.join('test_results.mpf')
        path = results.save(str(file), overwrite=True)
        assert file.check() is True

    def test_pickle_restore(self, results, temp_scratch):
        file = temp_scratch.join('test_results.mpf')
        path = results.save(str(file), overwrite=True)
        assert file.check() is True
        r = Results.restore(str(file))
        assert r.searchfilter == results.searchfilter


class TestResultsConvertTool(object):

    @pytest.mark.parametrize('results', [('nsa.z < 0.1 and haflux > 25')], indirect=True)
    @pytest.mark.parametrize('objtype, tool',
                             [('cube', Cube), ('maps', Maps), ('spaxel', Spaxel),
                              ('modelcube', ModelCube)])
    def test_convert_success(self, results, objtype, tool, exporigin):
        if config.release == 'MPL-4' and objtype == 'modelcube':
            pytest.skip('no modelcubes in mpl-4')

        results.convertToTool(objtype, limit=1)
        assert results.objects is not None
        assert isinstance(results.objects, list) is True
        assert isinstance(results.objects[0], tool) is True
        if objtype != 'spaxel':
            assert results.mode == results.objects[0].mode

    @pytest.mark.parametrize('objtype, error, errmsg',
                             [('modelcube', MarvinError, "ModelCube requires at least dapver='2.0.2'"),
                              ('spaxel', AssertionError, 'Parameters must include spaxelprop.x and y in order to convert to Marvin Spaxel')],
                             ids=['mcminrelease', 'nospaxinfo'])
    def test_convert_failures(self, results, objtype, error, errmsg):
        if config.release > 'MPL-4' and objtype == 'modelcube':
            pytest.skip('modelcubes in post mpl-4')

        with pytest.raises(error) as cm:
            results.convertToTool(objtype, limit=1)
        assert cm.type == error
        assert errmsg in str(cm.value)


