#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-04-08 14:31:34
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-04-08 14:31:34 by Brian Cherinka
    Last Modified On: 2016-04-08 14:31:34 by Brian

'''
from __future__ import division, print_function

import os

import numpy as np
from brain.utils.general.general import convertIvarToErr
from flask import Blueprint, jsonify, render_template, request
from flask import session as current_session
from flask_classful import route

import marvin
from marvin.api.base import arg_validate as av
from marvin.core import marvin_pickle
from marvin.core.caching_query import FromCache
from marvin.core.exceptions import MarvinError
from marvin.tools.cube import Cube
from marvin.utils.datamodel.dap import datamodel
from marvin.utils.general.general import (_db_row_to_dict, convertImgCoords, getDapRedux,
                                          getDefaultMapPath, parseIdentifier, target_status,
                                          get_manga_image)
from marvin.utils.general.maskbit import Maskbit
from marvin.web.controllers import BaseWebView
from marvin.web.extensions import cache


try:
    from sdss_access.path import Path
except ImportError as e:
    Path = None

galaxy = Blueprint("galaxy_page", __name__)


def getWebSpectrum(cube, x, y, xyorig=None, byradec=False):
    ''' Get and format a spectrum for the web '''
    webspec = None
    default_bintype = datamodel[cube.release].default_bintype.name
    has_models = cube.data.has_modelspaxels(name=default_bintype) if hasattr(cube.data, 'has_modelspaxels') else False

    # set the spaxel kwargs
    kwargs = {'xyorig': xyorig, 'maps': False}
    kwargs['modelcube'] = has_models
    if byradec:
        kwargs.update({'ra': x, 'dec': y})
    else:
        kwargs.update({'x': x, 'y': y})

    # get the spaxel
    try:
        spaxel = cube.getSpaxel(**kwargs)
    except Exception as e:
        specmsg = 'Could not get spaxel: {0}'.format(e)
    else:
        # get error and wavelength
        error = convertIvarToErr(spaxel.flux.ivar)
        wave = spaxel.flux.wavelength

        # try to get the model flux
        try:
            modelfit = spaxel.full_fit
        except Exception as e:
            modelfit = None

        # make input array for Dygraph
        if not isinstance(modelfit, type(None)):
            webspec = [[wave.value[i], [s, error[i]], [modelfit.value[i], 0.0]] for i, s in enumerate(spaxel.flux.value)]
        else:
            webspec = [[wave.value[i], [s, error[i]]] for i, s in enumerate(spaxel.flux.value)]

        specmsg = "Spectrum in Spaxel (j, i)=({2},{3}) at RA, Dec = ({0}, {1})".format(spaxel.ra, spaxel.dec, spaxel.x, spaxel.y)

    return webspec, specmsg


def getWebMap(cube, parameter='emline_gflux', channel='ha_6564',
              bintype=None, template=None):
    ''' Get and format a map for the web '''
    if channel:
        name = '{0}_{1}'.format(parameter.lower(), channel)
    else:
        name = '{0}'.format(parameter.lower())

    webmap = None
    try:
        maps = cube.getMaps(plateifu=cube.plateifu, mode='local',
                            bintype=bintype, template=template)
        data = maps.getMap(parameter, channel=channel)

        # correct the stellar_sigma or emline_gsigma maps
        if parameter == 'stellar_sigma' or parameter == 'emline_gsigma':
            try:
                data = data.inst_sigma_correction()
            except MarvinError as e:
                pass
            else:
                name = 'Corrected {0}'.format(name)

    except Exception as e:
        mapmsg = 'Could not get map: {0}'.format(e)
    else:
        vals = data.value
        ivar = data.ivar
        mask = data.mask
        webmap = {'values': [it.tolist() for it in data.value],
                  'ivar': [it.tolist() for it in data.ivar] if data.ivar is not None else None,
                  'mask': [it.tolist() for it in data.mask] if data.mask is not None else None}
        mapmsg = "{0}: {1}-{2}".format(name, maps.bintype, maps.template)
    return webmap, mapmsg


def buildMapDict(cube, params, dapver, bintemp=None):
    ''' Build a list of dictionaries of maps

    params - list of string parameter names in form of category_channel

        NOT GENERALIZED
    '''
    # split the bintemp
    if bintemp:
        bintype, temp = bintemp.split('-', 1)
    else:
        bintype, temp = (None, None)

    mapdict = []
    params = params if isinstance(params, list) else [params]

    for param in params:
        param = str(param)
        try:
            parameter, channel = param.split(':')
        except ValueError as e:
            parameter, channel = (param, None)
        webmap, mapmsg = getWebMap(cube, parameter=parameter, channel=channel,
                                   bintype=bintype, template=temp)

        plotparams = datamodel[dapver].get_plot_params(prop=parameter)
        mask = Maskbit('MANGA_DAPPIXMASK')
        baddata_labels = [it for it in plotparams['bitmasks'] if it != 'NOCOV']
        baddata_bits = {it.lower(): int(mask.labels_to_bits(it)[0]) for it in baddata_labels}
        plotparams['bits'] = {'nocov': int(mask.labels_to_bits('NOCOV')[0]),
                              'badData': baddata_bits}
        mapdict.append({'data': webmap, 'msg': mapmsg, 'plotparams': plotparams})

    anybad = [m['data'] is None for m in mapdict]
    if any(anybad):
        bad_params = ', '.join([p for i, p in enumerate(params) if anybad[i]])
        raise MarvinError('Could not get map for: {0}.  Please select another.'.format(bad_params),
                          ignore_git=True)

    return mapdict


def make_nsa_dict(nsa, cols=None):
    ''' Make/rearrange the nsa dictionary of values '''

    # get columns
    if not cols:
        cols = [k for k in nsa.keys() if 'stokes' not in k]
        cols.sort()

    # make dictionary
    nsadict = {c: nsa[c] for c in cols}
    absmag = nsadict.get('elpetro_absmag', None)
    mtol = nsadict.get('elpetro_mtol', None)
    if absmag:
        nsadict.update({'elpetro_absmag_i': nsadict['elpetro_absmag'][5]})
    if mtol:
        nsadict.update({'elpetro_mtol_i': nsadict['elpetro_mtol'][5]})

    # check for logmass
    if 'elpetro_logmass' not in nsadict:
        if 'elpetro_mass' in nsadict:
            nsadict.update({'elpetro_logmass': np.log10(nsadict['elpetro_mass'])})
            cols.append('elpetro_logmass')

    cols.append('elpetro_mtol_i')
    cols.append('elpetro_absmag_i')
    cols.sort()

    return nsadict, cols


def get_nsa_dict(name, drpver, makenew=None):
    ''' Gets a NSA dictionary from a pickle or a query '''
    nsapath = os.environ.get('MANGA_SCRATCH_DIR', None)
    if nsapath and os.path.isdir(nsapath):
        nsapath = nsapath
    else:
        nsapath = os.path.expanduser('~')

    nsaroot = os.path.join(nsapath, 'nsa_pickles')
    if not os.path.isdir(nsaroot):
        os.makedirs(nsaroot)

    picklename = '{0}.pickle'.format(name)
    nsapickle_file = os.path.join(nsaroot, picklename)
    if os.path.isfile(nsapickle_file):
        nsadict = marvin_pickle.restore(nsapickle_file)
    else:
        # make from scratch from db
        marvindb = marvin.marvindb
        session = marvindb.session
        sampledb = marvindb.sampledb
        allnsa = session.query(sampledb.NSA, marvindb.datadb.Cube.plateifu).\
            join(sampledb.MangaTargetToNSA, sampledb.MangaTarget,
                 marvindb.datadb.Cube, marvindb.datadb.PipelineInfo,
                 marvindb.datadb.PipelineVersion, marvindb.datadb.IFUDesign).\
            filter(marvindb.datadb.PipelineVersion.version == drpver).options(FromCache(name)).all()
        nsadict = [(_db_row_to_dict(n[0], remove_columns=['pk', 'catalogue_pk']), n[1]) for n in allnsa]

        # write a new NSA pickle object
        if makenew:
            marvin_pickle.save(nsadict, path=nsapickle_file, overwrite=True)

    return nsadict


def remove_nans(datadict):
    ''' Removes objects with nan values from the NSA sample dictionary '''

    # collect total unique indices of nan objects
    allnans = np.array([])
    for key, vals in datadict.items():
        if key != 'plateifu':
            naninds = np.where(np.isnan(vals))[0]
            allnans = np.append(allnans, naninds)
    allnans = list(set(allnans))
    # delete those targets from all items in the dictionary
    for key, vals in datadict.items():
        datadict[key] = np.delete(np.asarray(vals), allnans).tolist()

    return datadict


class Galaxy(BaseWebView):
    route_base = '/galaxy/'

    def __init__(self):
        ''' Initialize the route '''
        super(Galaxy, self).__init__('marvin-galaxy')
        self.galaxy = self.base.copy()
        self.galaxy['cube'] = None
        self.galaxy['image'] = ''
        self.galaxy['spectra'] = 'null'
        self.galaxy['maps'] = None
        self.galaxy['specmsg'] = None
        self.galaxy['mapmsg'] = None
        self.galaxy['toggleon'] = 'false'
        self.galaxy['nsamsg'] = None
        self.galaxy['hasnsa'] = False
        self.galaxy['nsachoices'] = {'1': {'y': 'z', 'x': 'elpetro_logmass', 'xtitle': 'Stellar Mass',
                                           'ytitle': 'Redshift', 'title': 'Redshift vs Stellar Mass'},
                                     '2': {'y': 'elpetro_absmag_g_r', 'x': 'elpetro_absmag_i', 'xtitle': 'AbsMag_i',
                                           'ytitle': 'Abs. g-r', 'title': 'Abs. g-r vs Abs. Mag i'}
                                     }
        # self.galaxy['nsachoices'] = {'1': {'y': 'z', 'x': 'sersic_mass', 'xtitle': 'Stellar Mass',
        #                                    'ytitle': 'Redshift', 'title': 'Redshift vs Stellar Mass'}}

        # cols = ['z', 'sersic_logmass', 'sersic_n', 'sersic_absmag', 'elpetro_mag_g_r', 'elpetro_th50_r']
        self.galaxy['nsaplotcols'] = ['z', 'elpetro_logmass', 'sersic_n', 'elpetro_absmag_i', 'elpetro_absmag_g_r',
                                      'elpetro_th50_r', 'elpetro_absmag_u_r', 'elpetro_absmag_i_z', 'elpetro_ba',
                                      'elpetro_phi', 'elpetro_mtol_i', 'elpetro_th90_r']

    def before_request(self, *args, **kwargs):
        ''' Do these things before a request to any route '''
        super(Galaxy, self).before_request(*args, **kwargs)
        self.reset_dict(self.galaxy, exclude=['nsachoices', 'nsaplotcols'])

    def index(self):
        ''' Main galaxy page '''
        self.galaxy['error'] = 'Not all there are you...  Try adding a plate-IFU or manga-ID to the end of the address.'
        return render_template("galaxy.html", **self.galaxy)

    def get(self, galid):
        ''' Retrieve info for a given cube '''

        # determine type of galid
        args = av.manual_parse(self, request, use_params='galaxy')
        self.galaxy['id'] = args['galid']
        self.galaxy['latest_dr'] = self._release.lower() if 'DR' in self._release else marvin.config._get_latest_release(dr_only=True).lower()
        idtype = parseIdentifier(galid)
        if idtype in ['plateifu', 'mangaid']:
            # set plateifu or mangaid
            self.galaxy['idtype'] = idtype
            galaxyid = {self.galaxy['idtype']: galid, 'release': self._release}

            # Get cube
            try:
                cube = Cube(**galaxyid)
            except MarvinError as e:
                self.galaxy['cube'] = None
                errmsg = 'MarvinError: {0}'.format(e)

                # check target status and fine-tune the error message
                if idtype == 'mangaid':
                    status = target_status(galid, drpver=self._drpver)
                    if status == 'not yet observed':
                        errmsg = '{0} is a valid target but has not yet been observed'.format(galid)
                    elif status == 'not valid target':
                        errmsg = '{0} is not valid MaNGA target.  Check your syntax'.format(galid)

                self.galaxy['error'] = errmsg
                return render_template("galaxy.html", **self.galaxy)
            else:
                dm = datamodel[self._dapver]
                self.galaxy['cube'] = cube
                self.galaxy['daplink'] = getDapRedux(release=self._release)
                # get SAS url links to cube, rss, maps, image
                if Path:
                    is_public = 'DR' in self._release
                    path_release = self._release.lower() if is_public else None
                    sdss_path = Path(public=is_public, release=path_release)
                    self.galaxy['image'] = get_manga_image(cube)
                    cubelink = sdss_path.url('mangacube', drpver=cube._drpver, plate=cube.plate, ifu=cube.ifu)
                    rsslink = sdss_path.url('mangarss', drpver=cube._drpver, plate=cube.plate, ifu=cube.ifu)
                    daptype = "{0}-{1}".format(dm.default_bintype, dm.default_template)
                    maplink = getDefaultMapPath(release=self._release, plate=cube.plate, ifu=cube.ifu, daptype=daptype, mode='MAPS')
                    mclink = getDefaultMapPath(release=self._release, plate=cube.plate, ifu=cube.ifu, daptype=daptype, mode='LOGCUBE')
                    self.galaxy['links'] = {'cube': cubelink, 'rss': rsslink, 'map': maplink, 'mc': mclink}
                else:
                    self.galaxy['image'] = cube.data.image

            # Get the initial spectrum
            if cube:
                dm = datamodel[self._dapver]
                daplist = [p.full(web=True) for p in dm.properties]
                dapdefaults = dm.get_default_mapset()
                self.galaxy['cube'] = cube
                self.galaxy['toggleon'] = current_session.get('toggleon', 'false')
                self.galaxy['cubehdr'] = cube.header
                self.galaxy['quality'] = ('DRP3QUAL', cube.quality_flag.mask, cube.quality_flag.labels)
                self.galaxy['mngtarget'] = {'bits': [it.mask for it in cube.target_flags if it.mask != 0],
                                            'labels': [it.labels for it in cube.target_flags if len(it.labels) > 0],
                                            'names': [''.join(('MNGTARG', it.name[-1])) for it in cube.target_flags if it.mask != 0]}

                # make the nsa dictionary
                hasnsa = cube.nsa is not None
                self.galaxy['hasnsa'] = hasnsa
                if hasnsa:
                    cols = self.galaxy.get('nsaplotcols')
                    nsadict, nsacols = make_nsa_dict(cube.nsa)
                    nsatmp = [nsacols.pop(nsacols.index(i)) for i in cols if i in nsacols]
                    nsatmp.extend(nsacols)
                    self.galaxy['nsacols'] = nsatmp
                    self.galaxy['nsadict'] = nsadict

                self.galaxy['dapmaps'] = daplist
                self.galaxy['dapmapselect'] = current_session.get('selected_dapmaps', dapdefaults)
                dm = datamodel[self._dapver]
                self.galaxy['dapbintemps'] = dm.get_bintemps(db_only=True)
                if 'bintemp' not in current_session:
                    current_session['bintemp'] = '{0}-{1}'.format(dm.get_bintype(), dm.get_template())

                # TODO - make this general - see also search.py for querystr
                self.galaxy['cubestr'] = ("<html><samp>from marvin.tools.cube import Cube<br>cube = \
                    Cube(plateifu='{0}')<br># access the header<br>cube.header<br># get NSA data<br>\
                    cube.nsa<br></samp></html>".format(cube.plateifu))

                self.galaxy['spaxelstr'] = ("<html><samp>from marvin.tools.cube import Cube<br>cube = \
                    Cube(plateifu='{0}')<br># get a spaxel by slicing cube[i,j]<br>spaxel=cube[16, 16]<br>flux = \
                    spaxel.flux<br>wave = flux.wavelength<br>ivar = flux.ivar<br>mask = \
                    flux.mask<br>flux.plot()<br></samp></html>".format(cube.plateifu))

                self.galaxy['mapstr'] = ("<html><samp>from marvin.tools.maps import Maps<br>maps = \
                    Maps(plateifu='{0}')<br>print(maps)<br># get an emission \
                    line map<br>haflux = maps.emline_gflux_ha_6564<br>values = \
                    haflux.value<br>ivar = haflux.ivar<br>mask = haflux.mask<br>haflux.plot()<br>\
                    </samp></html>".format(cube.plateifu))
        else:
            self.galaxy['error'] = 'Error: Galaxy ID {0} must either be a Plate-IFU, or MaNGA-Id designation.'.format(galid)
            return render_template("galaxy.html", **self.galaxy)

        return render_template("galaxy.html", **self.galaxy)

    @route('/initdynamic/', methods=['POST'], endpoint='initdynamic')
    def initDynamic(self):
        ''' Route to run when the dynamic toggle is initialized
            This creates the web spectrum and dap heatmaps
        '''

        # get the form parameters
        args = av.manual_parse(self, request, use_params='galaxy', required='plateifu')

        # datamodel
        dm = datamodel[self._dapver]

        # turning toggle on
        nowebsession = marvin.config._custom_config.get('no_web_session', None)
        if not nowebsession: 
            current_session['toggleon'] = args.get('toggleon')

        # get the cube
        cubeinputs = {'plateifu': args.get('plateifu'), 'release': self._release}
        cube = Cube(**cubeinputs)
        output = {'specstatus': -1, 'mapstatus': -1}

        # get web spectrum
        webspec, specmsg = getWebSpectrum(cube, cube.ra, cube.dec, byradec=True)
        daplist = [p.full(web=True) for p in dm.properties]
        dapdefaults = dm.get_default_mapset()

        # select any DAP maps and bin-template from the session
        selected_bintemp = current_session.get('bintemp', None)
        selected_maps = current_session.get('selected_dapmaps', dapdefaults)

        # build the uber map dictionary
        try:
            mapdict = buildMapDict(cube, selected_maps, self._dapver, bintemp=selected_bintemp)
            mapmsg = None
        except Exception as e:
            mapdict = [{'data': None, 'msg': 'Error', 'plotparams': None} for m in dapdefaults]
            mapmsg = 'Error getting maps: {0}'.format(e)
        else:
            output['mapstatus'] = 1

        if not webspec:
            output['error'] = 'Error: {0}'.format(specmsg)
        else:
            output['specstatus'] = 1

        output['image'] = get_manga_image(cube)
        output['spectra'] = webspec
        output['specmsg'] = specmsg
        output['maps'] = mapdict
        output['mapmsg'] = mapmsg
        output['dapmaps'] = daplist
        print('selected maps', selected_maps)
        output['dapmapselect'] = selected_maps

        output['dapbintemps'] = dm.get_bintemps(db_only=True)
        if 'bintemp' not in current_session:
            current_session['bintemp'] = '{0}-{1}'.format(dm.get_bintype(), dm.get_template())

        # try to jsonify the result
        try:
            jsonout = jsonify(result=output)
        except Exception as e:
            jsonout = jsonify(result={'specstatus': -1, 'mapstatus': -1, 'error': '{0}'.format(e)})

        return jsonout

    @route('/getspaxel/', methods=['POST'], endpoint='getspaxel')
    def getSpaxel(self):
        args = av.manual_parse(self, request, use_params='galaxy', required=['plateifu', 'type'], makemulti=True)
        cubeinputs = {'plateifu': args.get('plateifu'), 'release': self._release}
        maptype = args.get('type', None)

        if maptype == 'optical':
            # for now, do this, but TODO - general processRequest to handle lists and not lists
            try:
                mousecoords = args.getlist('mousecoords[]', type=float)
            except Exception as e:
                mousecoords = None

            if mousecoords:
                pixshape = (args.get('imwidth', type=int), args.get('imheight', type=int))
                if (mousecoords[0] < 0 or mousecoords[0] > pixshape[0]) or (mousecoords[1] < 0 or mousecoords[1] > pixshape[1]):
                    output = {'specmsg': 'Error: requested pixel coords are outside the image range.', 'status': -1}
                    self.galaxy['error'] = output['specmsg']
                else:
                    # TODO - generalize image file sas_url to filesystem switch, maybe in sdss_access
                    infile = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'), args.get('image').split('redux/')[1])
                    arrcoords = convertImgCoords(mousecoords, infile, to_radec=True)

                    cube = Cube(**cubeinputs)
                    webspec, specmsg = getWebSpectrum(cube, arrcoords[0], arrcoords[1], byradec=True)
                    if not webspec:
                        self.galaxy['error'] = 'Error: {0}'.format(specmsg)
                        status = -1
                    else:
                        status = 1
                    msg = 'gettin some spaxel at RA/Dec {0}'.format(arrcoords)
                    output = {'message': msg, 'specmsg': specmsg, 'spectra': webspec, 'status': status}
            else:
                output = {'specmsg': 'Error getting mouse coords', 'status': -1}
                self.galaxy['error'] = output['specmsg']
        elif maptype == 'heatmap':
            # grab spectrum based on (x, y) coordinates
            x = args.get('x', None, type=int)
            y = args.get('y', None, type=int)
            if all([x, y]):
                cube = Cube(**cubeinputs)
                webspec, specmsg = getWebSpectrum(cube, x, y, xyorig='lower')
                msg = 'gettin some spaxel with (x={0}, y={1})'.format(x, y)
                if not webspec:
                    self.galaxy['error'] = 'Error: {0}'.format(specmsg)
                    status = -1
                else:
                    status = 1
                output = {'message': msg, 'specmsg': specmsg, 'spectra': webspec, 'status': status}
            else:
                output = {'specmsg': 'Error: X or Y not specified for map', 'status': -1}
                self.galaxy['error'] = output['specmsg']
        else:
            output = {'specmsg': 'Error: No maptype specified in request', 'status': -1}
            self.galaxy['error'] = output['specmsg']

        return jsonify(result=output)

    @route('/updatemaps/', methods=['POST'], endpoint='updatemaps')
    def updateMaps(self):
        args = av.manual_parse(self, request, use_params='galaxy', required=['plateifu', 'bintemp', 'params[]'], makemulti=True)
        #self._drpver, self._dapver, self._release = parseSession()
        cubeinputs = {'plateifu': args.get('plateifu'), 'release': self._release}
        params = args.getlist('params[]', type=str)
        bintemp = args.get('bintemp', None, type=str)

        # update the session variables
        current_session['bintemp'] = bintemp
        current_session['selected_dapmaps'] = params

        # get cube (self.galaxy['cube'] does not work)
        try:
            cube = Cube(**cubeinputs)
        except Exception as e:
            cube = None
        # Try to make the web maps
        if not cube:
            output = {'mapmsg': 'No cube found', 'maps': None, 'status': -1}
        elif not params:
            output = {'mapmsg': 'No parameters selected', 'maps': None, 'status': -1}
        else:
            try:
                mapdict = buildMapDict(cube, params, self._dapver, bintemp=bintemp)
            except Exception as e:
                output = {'mapmsg': str(e), 'status': -1, 'maps': None}
            else:
                output = {'mapmsg': None, 'status': 1, 'maps': mapdict}
        return jsonify(result=output)

    @cache.cached(timeout=300, key_prefix='init_nsa')
    @route('/initnsaplot/', methods=['POST'], endpoint='initnsaplot')
    def init_nsaplot(self):
        args = av.manual_parse(self, request, use_params='galaxy', required='plateifu')
        #self._drpver, self._dapver, self._release = parseSession()
        cubeinputs = {'plateifu': args.get('plateifu'), 'release': self._release}

        # get the default nsa choices
        nsachoices = self.galaxy.get('nsachoices', None)
        if not nsachoices:
            nsachoices = {'1': {'y': 'z', 'x': 'elpetro_logmass', 'xtitle': 'Stellar Mass',
                                'ytitle': 'Redshift', 'title': 'Redshift vs Stellar Mass'},
                          '2': {'y': 'elpetro_absmag_g_r', 'x': 'elpetro_absmag_i', 'xtitle': 'AbsMag_i',
                                'ytitle': 'Abs. g-r', 'title': 'Abs. g-r vs Abs. Mag i'}}

        # get cube (self.galaxy['cube'] does not work)
        try:
            cube = Cube(**cubeinputs)
        except Exception as e:
            cube = None

        # get some nsa params
        if not cube:
            output = {'nsamsg': 'No cube found', 'nsa': None, 'status': -1}
        else:
            # get the galaxy nsa parameters
            cols = self.galaxy.get('nsaplotcols')
            try:
                nsadict, nsacols = make_nsa_dict(cube.nsa)
                nsa = {args.get('plateifu'): nsadict}
            except Exception as e:
                output = {'nsamsg': str(e), 'status': -1, 'nsa': None}
            else:
                # get the sample nsa parameters
                try:
                    nsacache = 'nsa_{0}'.format(self._release.lower().replace('-', ''))
                    nsadict = get_nsa_dict(nsacache, self._drpver)
                except Exception as e:
                    output = {'nsamsg': 'Failed to retrieve sample NSA: {0}'.format(e), 'status': -1, 'nsa': nsa, 'nsachoices': nsachoices}
                else:
                    #nsadict = [(_db_row_to_dict(n[0], remove_columns=['pk', 'catalogue_pk']), n[1]) for n in allnsa]
                    nsasamp = {c: [n[0][c.split('_i')[0]][5] if 'absmag_i' in c or 'mtol_i' in c else n[0][c] for n in nsadict] for c in cols}
                    nsasamp['plateifu'] = [n[1] for n in nsadict]
                    nsasamp = remove_nans(nsasamp)
                    nsa['sample'] = nsasamp
                    output = {'nsamsg': None, 'status': 1, 'nsa': nsa, 'nsachoices': nsachoices, 'nsaplotcols': cols}
        return jsonify(result=output)


Galaxy.register(galaxy)
