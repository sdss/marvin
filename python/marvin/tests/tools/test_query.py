# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-25 10:11:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-19 23:36:43

from __future__ import absolute_import, division, print_function

from imp import reload
import pytest

import marvin
from marvin import config
from marvin.core.exceptions import MarvinError
from marvin.tests import marvin_test_if
from marvin.tests.conftest import set_the_config
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube
from marvin.tools.query import Query, doQuery
from marvin.tools.spaxel import Spaxel


@pytest.fixture(scope='function', autouse=True)
def allow_dap(monkeypatch):
    monkeypatch.setattr(config, '_allow_DAP_queries', True)
    global Query, doQuery
    reload(marvin.tools.query)
    from marvin.tools.query import Query, doQuery


class TestDoQuery(object):

    def test_success(self, release, mode):
        q, r = doQuery(search_filter='nsa.z < 0.1', release=release, mode=mode)
        assert q is not None
        assert r is not None


class TestQueryVersions(object):

    def test_versions(self, query, release, versions):
        drpver, dapver = versions
        assert query.release == release
        assert query._drpver == drpver
        assert query._dapver == dapver


class TestQuerySearches(object):

    @pytest.mark.parametrize('query, joins',
                             [('nsa.z < 0.1', ['nsa', 'drpalias']),
                              ('emline_gflux_ha_6564 > 25', ['spaxelprop', 'dapalias']),
                              ('nsa.z < 0.1 and emline_gflux_ha_6564 > 25', ['nsa', 'spaxelprop', 'drpalias', 'dapalias'])],
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
                              ('emline_gflux_ha_6564 > 25', ['emline_gflux_ha_6564', 'spaxelprop.x', 'spaxelprop.y', 'bintype.name', 'template.name'])],
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

        set_the_config(config.release)

        with pytest.raises((KeyError, MarvinError)) as cm:
            query = Query(search_filter=badquery, mode=expmode)
            res = query.run()
        assert cm.type == KeyError or cm.type == MarvinError
        assert errmsg in str(cm.value)

    @pytest.mark.parametrize('query, sfilter',
                             [('nsa.z < 0.1', 'nsa.z < 0.1'),
                              ('absmag_g_r > -1', 'absmag_g_r > -1'),
                              ('emline_gflux_ha_6564 > 25', 'emline_gflux_ha_6564 > 25'),
                              ('npergood(emline_gflux_ha_6564 > 5) > 20', 'npergood(emline_gflux_ha_6564 > 5) > 20'),
                              ('nsa.z < 0.1 and emline_gflux_ha_6564 > 25', 'nsa.z < 0.1 and emline_gflux_ha_6564 > 25')],
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
        query = Query(search_filter=query.search_filter, mode=query.mode, sort=sortparam, order=order)
        res = query.run()
        if order == 'asc':
            redshift = data['1'][-1]
        else:
            redshift = data['last'][-1]
        assert res.results['z'][0] == redshift


class TestQueryShow(object):

    @pytest.mark.parametrize('query, show, exp',
                             [('nsa.z < 0.1', 'query', 'SELECT mangadatadb.cube.mangaid'),
                              ('nsa.z < 0.1', 'joins', "['ifudesign', 'manga_target', 'manga_target_to_nsa', 'nsa']"),
                              ('nsa.z < 0.1', 'filter', 'mangasampledb.nsa.z < 0.1')], indirect=['query'])
    def test_show(self, query, show, exp, capsys):
        sql = query.show(show)
        if query.mode == 'remote':
            assert sql == query.search_filter
        else:
            assert exp in sql or exp == sql.strip('\n')


# class TestQueryReturnParams(object):

#     @pytest.mark.parametrize('query', [('nsa.z < 0.1')], indirect=True)
#     @pytest.mark.parametrize('rps', [(['g_r']), (['cube.ra', 'cube.dec']), (['haflux'])])
#     def test_success(self, query, rps):
#         query = Query(search_filter=query.search_filter, return_params=rps, mode=query.mode)
#         params = query._remote_params['params'].split(',') if query.mode == 'remote' else query.params
#         #assert 'nsa.z' in params
#         names = [query._marvinform._param_form_lookup._nameShortcuts[r] for r in rps]
#         assert set(names).issubset(set(query.params))
#         res = query.run()
#         assert all([p in res.columns for p in rps]) is True
#         #assert set(rps).issubset(set(query.params))
#         #assert set(rps).issubset(set(res.paramtocol.keys()))

#     @pytest.mark.parametrize('query', [('nsa.z < 0.1')], indirect=True)
#     @pytest.mark.parametrize('rps, errmsg',
#                              [('hello', 'does not match any column.'),
#                               ('name', 'name matches multiple parameters')],
#                              ids=['nomatch', 'multiple_entries'])
#     def test_badparams(self, query, expmode, rps, errmsg):
#         # set error type based on query mode
#         if expmode == 'remote':
#             error = MarvinError
#         else:
#             error = KeyError

#         with pytest.raises(error) as cm:
#             query = Query(search_filter=query.search_filter, return_params=[rps], mode=query.mode)
#             res = query.run()
#         assert cm.type == error
#         assert errmsg in str(cm.value)

#     @pytest.mark.parametrize('query', [('nsa.z < 0.1')], indirect=True)
#     @pytest.mark.parametrize('rps', [(['absmag_g_r', 'cube.plate', 'cube.plateifu'])])
#     def test_skipdefault(self, query, rps):
#         query = Query(search_filter=query.search_filter, return_params=rps, mode=query.mode)
#         params = query._remote_params['params'].split(',') if query.mode == 'remote' else query.params
#         assert len(query.return_params) == len(rps)
#         assert len(params) == 5
#         res = query.run()
#         assert len(res.returnparams) == len(rps)
#         assert len(res.columns) == 5


class TestQueryReturnType(object):

    @pytest.mark.parametrize('query', [('cube.mangaid == 1-209232 and emline_gflux_ha_6564 > 25')], indirect=True)
    @pytest.mark.parametrize('objtype, tool',
                             [('cube', Cube), ('maps', Maps), ('spaxel', Spaxel),
                              ('modelcube', ModelCube)])
    @marvin_test_if(mark='skip', query={'release': ['MPL-4']})
    def test_get_success(self, query, objtype, tool):
        if query.mode == 'remote' and config.db is None:
            pytest.skip('skipping weird case where nodb, remote mode tried to load a local file')
        if config.release == 'MPL-4' and objtype == 'modelcube':
            pytest.skip('no modelcubes in mpl-4')

        query = Query(search_filter=query.search_filter, return_type=objtype, mode=query.mode, release=query.release)
        res = query.run()
        assert res.objects is not None
        assert len(res.results) == len(res.objects)
        assert isinstance(res.objects[0], tool) is True

    @pytest.mark.parametrize('query', [('nsa.z < 0.1')], indirect=True)
    @pytest.mark.parametrize('objtype, errmsg',
                             [('noncube', 'Query return_type must be either cube, spaxel, maps, modelcube, rss')])
    @marvin_test_if(mark='skip', query={'mode': ['remote']})
    def test_badreturntype(self, query, objtype, errmsg):
        with pytest.raises(AssertionError) as cm:
            query = Query(search_filter=query.search_filter, return_type=objtype, mode=query.mode)
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
        assert query.search_filter == sfilter


@pytest.mark.skip(reason="no spaxel queries causes this to fail since parameter counts are now off")
class TestQueryParams(object):

    @pytest.mark.parametrize('paramdisplay', [('all'), ('best')])
    def test_getparams(self, query, paramdisplay):
        params = query.get_available_params(paramdisplay, release=query.release)
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


@pytest.fixture(scope='class')
def lquery(request):
    searchfilter = request.param if hasattr(request, 'param') else None
    q = Query(search_filter=searchfilter, mode='local')
    yield q
    q = None


@pytest.fixture(scope='class')
def rquery(request):
    config.forceDbOff()
    searchfilter = request.param if hasattr(request, 'param') else None
    q = Query(search_filter=searchfilter, mode='remote')
    yield q
    q = None
    config.forceDbOn()

#
# Below here begins a low refactor of the query tests
#

class TestQueryLocal(object):
    mode = 'local'
    sf = 'nsa.z < 0.1'

    @pytest.mark.parametrize('rps', [(['g_r']), (['cube.ra', 'cube.dec']), (['emline_gflux_ha_6564'])], ids=['g-r', 'radec', 'haflux'])
    def test_return_params(self, rps):
        base = ['cube.ra', 'cube.dec']
        query = Query(search_filter=self.sf, mode=self.mode, return_params=base + rps)
        reals = [query._marvinform._param_form_lookup.get_real_name(r) for r in rps]
        assert set(reals).issubset(set(query.params))
        assert set(reals).issubset(query.return_params)

    @pytest.mark.parametrize('rps, errmsg',
                             [('hello', 'does not match any column.'),
                              ('name', 'name matches multiple parameters')],
                             ids=['nomatch', 'multiple_entries'])
    def test_bad_returnparams(self, rps, errmsg):
        with pytest.raises(KeyError) as cm:
            query = Query(search_filter=self.sf, return_params=[rps], mode=self.mode)
        assert cm.type == KeyError
        assert errmsg in str(cm.value)

    @pytest.mark.parametrize('rps', [(['absmag_g_r', 'cube.plate', 'cube.plateifu'])])
    def test_skipdefault(self, rps):
        query = Query(search_filter=self.sf, return_params=rps, mode=self.mode)
        assert len(query.return_params) == len(rps)
        assert len(query.params) == 5
        assert query.params.count('cube.plateifu') == 1

    @pytest.mark.parametrize('sf, rps, schema, tables, check',
                             [(None, None, 'sampledb', None, True),
                              (None, None, 'dapdb', None, False),
                              (None, 'stellar_vel', 'dapdb', None, True),
                              ('stellar_vel > 50', None, 'dapdb', 'spaxelprop', True),
                              (None, 'sb_1re', None, 'dapall', True),
                              ('stellar_z < 0.1', None, 'dapdb', 'dapall', True),
                              ('sb_1re > 0.2', None, 'dapdb', 'cube', False)
                              ],
                             ids=['nsaz', 'nodap', 'daprp', 'dap', 'dapallrp',
                                  'dapall', 'wrongtableinschema'])
    def test_checkfor(self, sf, rps, schema, tables, check):
        sf = sf or self.sf
        query = Query(search_filter=sf, return_params=rps, mode=self.mode)
        isin = query._check_for(query.params, schema=schema, tables=tables)
        assert isin == check

    @pytest.mark.parametrize('sf, rps, schema, tables, check',
                             [(None, None, None, ['cube', 'nsa'], True),
                              ('sb_1re > 0.2', None, None, 'dapall', False),
                              ('sb_1re > 0.2', None, 'dapdb', 'dapall', False),
                              ('sb_1re > 0.2', None, 'dapdb', ['dapall', 'bintype', 'template'], True),
                              (None, 'stellar_z', 'dapdb', ['dapall', 'bintype', 'template'], True)
                              ],
                             ids=['drp', 'noschema', 'onlydapall', 'good', 'goodrp'])
    def test_checkonly(self, sf, rps, schema, tables, check):
        sf = sf or self.sf
        query = Query(search_filter=sf, return_params=rps, mode=self.mode)
        isin = query._check_for(query.params, schema=schema, tables=tables, only=True)
        assert isin == check

    @pytest.mark.parametrize('sf, rps',
                             [(None, 'stellar_vel'),
                              ('emline_gflux_ha_6564 > 25', None),
                              ('stellar_z < 0.1 and stellar_vel > 50', None),
                              ('sb_1re > 0.2', 'stellar_vel')],
                             ids=['nodaprp', 'nodap', 'dapall', 'daprp'])
    def test_nodap(self, monkeypatch, sf, rps):
        sf = sf or self.sf
        monkeypatch.setattr(config, '_allow_DAP_queries', False)
        with pytest.raises(NotImplementedError) as cm:
            query = Query(search_filter=sf, return_params=rps, mode=self.mode)
        assert 'DAP spaxel queries are disabled in this version.' in str(cm)

    @pytest.mark.parametrize('sf',
                             [('radial(232.5447, 48.6902, 1)'),
                              ('npergood(emline_gflux_ha_6564 > 5) > 20')],
                             ids=['radial', 'npergood'])
    def test_function_queries(self, sf):
        q = Query(search_filter=sf)
        r = q.run()
        assert r.results is not None

    @pytest.mark.parametrize('sf, qual, flags, count',
                             [(None, None, 'mangadatadb.cube', 5),
                              (None, ['BADFLUX'], 'DRP3QUAL', 0),
                              (None, ['BADFLUX', 'BADZ'], 'DRP3QUAL, DAPQUAL', 0),
                              ('cube.quality = 0', None, 'DRP3QUAL', 5),
                              ('cube.quality & 256', None, 'DRP3QUAL', 0),
                              (None, ['FORESTAR'], 'DAPQUAL', 40),
                              ('cube.quality & ~0', None, 'DRP3QUAL', 0)],
                             ids=['all', 'drp', 'drpdap', 'goodqual',
                                  'badflux', 'forestar', 'not0'])
    def test_quality_queries(self, sf, qual, flags, count):
        q = Query(search_filter=sf, quality=qual, release='MPL-5')
        sql = q.show()
        for flag in flags.split(','):
            assert flag.strip() in sql
        r = q.run()
        assert r.results is not None
        assert r.totalcount == count

    @pytest.mark.parametrize('sf, targ, flags, count',
                             [(None, None, 'mangadatadb.cube', 3282),
                              (None, ['PRIMARY'], 'MNGT%RG1', 846),
                              (None, ['mwa', 'dwarf'], 'MNGT%RG3', 4)],
                             ids=['all', 'primary', 'anc'])
    def test_target_queries(self, sf, targ, flags, count):
        q = Query(search_filter=sf, targets=targ, release='MPL-4')
        sql = q.show()
        for flag in flags.split(','):
            assert flag.strip() in sql
        r = q.run()
        assert r.results is not None
        assert r.totalcount == count


class TestQueryAuto(object):

    @pytest.mark.parametrize('mode', [('local'), ('remote')])
    def test_mma(self, mode):
        if mode == 'remote':
            config.forceDbOff()
        q = Query()
        assert q.mode == mode


@pytest.fixture(scope='class')
def dboff():
    config.forceDbOff()
    yield True
    config.forceDbOn()


@pytest.mark.usefixtures("dboff")
class TestQueryRemote(object):
    mode = 'remote'
    sf = 'nsa.z < 0.1'

    @pytest.mark.parametrize('rps', [(['g_r']), (['cube.ra', 'cube.dec']),
                                     (['emline_gflux_ha_6564'])],
                             ids=['g-r', 'radec', 'haflux'])
    def test_return_params(self, rps):
        base = ['cube.ra', 'cube.dec']
        query = Query(search_filter=self.sf, mode=self.mode, return_params=base + rps)
        params = query._remote_params['returnparams'].split(',')
        assert set(rps).issubset(set(params))
