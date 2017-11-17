# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-25 10:11:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-11-17 12:54:33

from __future__ import print_function, division, absolute_import
from marvin.tools.query import Query, doQuery
from marvin.core.exceptions import MarvinError
from marvin import config
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.spaxel import Spaxel
from marvin.tools.modelcube import ModelCube
import pytest


class TestDoQuery(object):

    def test_success(self, release, mode):
        q, r = doQuery(searchfilter='nsa.z < 0.1', release=release, mode=mode)
        assert q is not None
        assert r is not None


class TestQueryVersions(object):

    def test_versions(self, query, release, versions):
        drpver, dapver = versions
        assert query._release == release
        assert query._drpver == drpver
        assert query._dapver == dapver


class TestQuerySearches(object):

    @pytest.mark.parametrize('query, joins',
                             [('nsa.z < 0.1', ['nsa', 'drpalias']),
                              ('haflux > 25', ['spaxelprop', 'dapalias']),
                              ('nsa.z < 0.1 and haflux > 25', ['nsa', 'spaxelprop', 'drpalias', 'dapalias'])],
                             ids=['drponly', 'daponly', 'drp_and_dap'],
                             indirect=['query'])
    def test_whereclause(self, query, joins):
        if query.mode == 'remote':
            res = query.run()
            for item in joins:
                assert item in str(res.query)
        else:
            for item in joins:
                assert item in str(query.query.whereclause)

    @pytest.mark.parametrize('query, addparam',
                             [('nsa.z < 0.1', ['nsa.z']),
                              ('haflux > 25', ['emline_gflux_ha_6564', 'spaxelprop.x', 'spaxelprop.y', 'bintype.name', 'template.name'])],
                             indirect=['query'])
    def test_params(self, query, addparam):
        params = query.expdata['defaults'] + addparam
        res = query.run()
        assert set(params) == set(query.params)

    @pytest.mark.parametrize('badquery, errmsg',
                             [('nsa.hello < 0.1', 'nsa.hello does not match any column.'),
                              ('name = SPX', 'name matches multiple parameters in the lookup table'),
                              ('< 0.1', 'Your boolean expression contained a syntax error')],
                             ids=['nomatch', 'multiple_entries', 'syntax_error'])
    def test_bad_queries(self, expmode, badquery, errmsg):
        if expmode is None:
            pytest.skip('cannot use queries in local mode without a db')

        with pytest.raises(MarvinError) as cm:
            query = Query(searchfilter=badquery, mode=expmode)
            res = query.run()
        assert cm.type == MarvinError
        assert errmsg in str(cm.value)

    # Keeping this test for posterity
    # @pytest.mark.parametrize('query, allspax, table',
    #                          [('haflux > 25', False, 'cleanspaxelprop'),
    #                           ('haflux > 25', True, 'spaxelprop')],
    #                          ids=['allspax', 'cleanspax'],
    #                          indirect=['query'])
    # def test_spaxel_tables(self, query, expmode, allspax, table):
    #     table = table + config.release.split('-')[1] if '4' not in config.release else table
    #     print('creating new query')
    #     query = Query(searchfilter=query.searchfilter, allspaxels=allspax, mode=query.mode, release=query._release)
    #     if expmode == 'local':
    #         assert table in set(query.joins)
    #     else:
    #         res = query.run()
    #         assert table in res.query

    @pytest.mark.parametrize('query, sfilter',
                             [('nsa.z < 0.1', 'nsa.z < 0.1'),
                              ('absmag_g_r > -1', 'absmag_g_r > -1'),
                              ('haflux > 25', 'emline_gflux_ha_6564 > 25'),
                              ('npergood(emline_gflux_ha_6564 > 5) > 20', 'npergood(emline_gflux_ha_6564 > 5) > 20'),
                              ('nsa.z < 0.1 and haflux > 25', 'nsa.z < 0.1 and emline_gflux_ha_6564 > 25')],
                             indirect=['query'], ids=['nsaz', 'absgr', 'haflux', 'npergood', 'nsahaflux'])
    def test_success_queries(self, query, sfilter):
        res = query.run()
        count = query.expdata['queries'][sfilter]
        assert count['count'] == res.totalcount

    # @pytest.mark.parametrize('query, qmode',
    #                          [('nsa.z < 0.1', 'count'),
    #                           ('nsa.z < 0.1', 'first')],
    #                          indirect=['query'])
    # def test_qmodes(self, query, qmode):
    #     mycount = query.expdata['queries']['nsa.z < 0.1']['count']
    #     r = query.run(qmode)
    #     if qmode == 'count':
    #         assert r == mycount
    #     elif qmode == 'first':
    #         assert len(r.results) == 1
    #         assert r.count == 1


class TestQuerySort(object):

    @pytest.mark.parametrize('query, sortparam, order',
                             [('nsa.z < 0.1', 'z', 'asc'),
                              ('nsa.z < 0.1', 'nsa.z', 'desc')], indirect=['query'])
    def test_sort(self, query, sortparam, order):
        data = query.expdata['queries']['nsa.z < 0.1']['sorted']
        query = Query(searchfilter=query.searchfilter, mode=query.mode, sort=sortparam, order=order)
        res = query.run()
        if order == 'asc':
            redshift = data['1'][-1]
        else:
            redshift = data['last'][-1]
        assert res.results['z'][0] == redshift


class TestQueryShow(object):

    @pytest.mark.parametrize('query, show, exp',
                             [('nsa.z < 0.1', 'query', 'SELECT mangadatadb.cube.mangaid'),
                              ('nsa.z < 0.1', 'tables', "['ifudesign', 'manga_target', 'manga_target_to_nsa', 'nsa']"),
                              ('nsa.z < 0.1', 'joins', "['ifudesign', 'manga_target', 'manga_target_to_nsa', 'nsa']"),
                              ('nsa.z < 0.1', 'filter', 'mangasampledb.nsa.z < 0.1')], indirect=['query'])
    def test_show(self, query, show, exp, capsys):
        if query.mode == 'remote':
            exp = 'Cannot show full SQL query in remote mode, use the Results showQuery'
        query.show(show)
        out, err = capsys.readouterr()
        assert exp in out or exp == out.strip('\n')


class TestQueryReturnParams(object):

    @pytest.mark.parametrize('query', [('nsa.z < 0.1')], indirect=True)
    @pytest.mark.parametrize('rps', [(['g_r']), (['cube.ra', 'cube.dec']), (['haflux'])])
    def test_success(self, query, rps):
        query = Query(searchfilter=query.searchfilter, returnparams=rps, mode=query.mode)
        assert 'nsa.z' in query.params
        #assert set(rps).issubset(set(query.params))
        res = query.run()
        assert all([p in res.columns for p in rps]) is True
        #assert set(rps).issubset(set(query.params))
        #assert set(rps).issubset(set(res.paramtocol.keys()))

    @pytest.mark.parametrize('query', [('nsa.z < 0.1')], indirect=True)
    @pytest.mark.parametrize('rps, errmsg',
                             [('hello', 'does not match any column.'),
                              ('name', 'name matches multiple parameters')],
                             ids=['nomatch', 'multiple_entries'])
    def test_badparams(self, query, expmode, rps, errmsg):
        # set error type based on query mode
        if expmode == 'remote':
            error = MarvinError
        else:
            error = KeyError

        with pytest.raises(error) as cm:
            query = Query(searchfilter=query.searchfilter, returnparams=[rps], mode=query.mode)
            res = query.run()
        assert cm.type == error
        assert errmsg in str(cm.value)


class TestQueryReturnType(object):

    @pytest.mark.parametrize('query', [('cube.mangaid == 1-209232 and haflux > 25')], indirect=True)
    @pytest.mark.parametrize('objtype, tool',
                             [('cube', Cube), ('maps', Maps), ('spaxel', Spaxel),
                              ('modelcube', ModelCube)])
    def test_get_success(self, query, objtype, tool):
        if query.mode == 'remote' and config.db is None:
            pytest.skip('skipping weird case where nodb, remote mode tried to load a local file')
        if config.release == 'MPL-4' and objtype == 'modelcube':
            pytest.skip('no modelcubes in mpl-4')

        query = Query(searchfilter=query.searchfilter, returntype=objtype, mode=query.mode, release=query._release)
        res = query.run()
        assert res.objects is not None
        assert len(res.results) == len(res.objects)
        assert isinstance(res.objects[0], tool) is True

    @pytest.mark.parametrize('query', [('nsa.z < 0.1')], indirect=True)
    @pytest.mark.parametrize('objtype, errmsg',
                             [('noncube', 'Query returntype must be either cube, spaxel, maps, modelcube, rss')])
    def test_badreturntype(self, query, objtype, errmsg):
        with pytest.raises(AssertionError) as cm:
            query = Query(searchfilter=query.searchfilter, returntype=objtype, mode=query.mode)
        assert cm.type == AssertionError
        assert errmsg in str(cm.value)


class TestQueryPickling(object):

    @pytest.mark.parametrize('query', [('nsa.z < 0.1')], indirect=True)
    def test_pickle_save(self, temp_scratch, query):
        if query.mode == 'local':
            pytest.xfail('save cannot be run in local mode')
        file = temp_scratch.join('test_query.mpf')
        query.save(str(file))
        assert file.check() is True

    @pytest.mark.parametrize('query, sfilter', [('nsa.z < 0.1', 'nsa.z < 0.1')], indirect=['query'])
    def test_pickle_restore(self, temp_scratch, query, sfilter):
        if query.mode == 'local':
            pytest.xfail('save cannot be run in local mode')
        file = temp_scratch.join('test_query.mpf')
        query.save(str(file))
        assert file.check() is True
        query = None
        assert query is None
        query = Query.restore(str(file))
        assert query.searchfilter == sfilter


class TestQueryParams(object):

    @pytest.mark.parametrize('paramdisplay', [('all'), ('best')])
    def test_getparams(self, query, paramdisplay):
        params = query.get_available_params(paramdisplay)
        mydata = query.expdata['params'][paramdisplay]
        # counts and content
        if paramdisplay == 'best':
            assert mydata['count'] == sum([len(v) for v in params])
            assert set(mydata['subset']).issubset(set(params.list_params()))
            assert set(mydata).isdisjoint(set(params.list_params()))
        elif paramdisplay == 'all':
            assert mydata['count'] == len(params)
            assert set(mydata['subset']).issubset(set(params))


class TestQueryModes(object):

    @pytest.mark.parametrize('query', [('nsa.z < 0.1 and cube.plate == 8485')], indirect=True)
    def test_getmode(self, query, expmode):
        assert query.mode == expmode
        res = query.run()
        assert res.mode == expmode
        assert query.mode == res.mode
