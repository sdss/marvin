/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 16:49:00
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2018-11-17 14:36:53
*/

//
// Javascript Galaxy object handling JS things for a single galaxy
//
//jshint esversion: 6
'use strict';

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var SpaxelError = function (_Error) {
    _inherits(SpaxelError, _Error);

    function SpaxelError(message) {
        _classCallCheck(this, SpaxelError);

        var _this2 = _possibleConstructorReturn(this, (SpaxelError.__proto__ || Object.getPrototypeOf(SpaxelError)).call(this, message));

        _this2.message = message;
        _this2.name = 'SpaxelError';
        _this2.status = -1;
        return _this2;
    }

    return SpaxelError;
}(Error);

var MapError = function (_Error2) {
    _inherits(MapError, _Error2);

    function MapError(message) {
        _classCallCheck(this, MapError);

        var _this3 = _possibleConstructorReturn(this, (MapError.__proto__ || Object.getPrototypeOf(MapError)).call(this, message));

        _this3.message = message;
        _this3.name = 'MapError';
        _this3.status = -1;
        return _this3;
    }

    return MapError;
}(Error);

var Galaxy = function () {

    // Constructor
    function Galaxy(plateifu, toggleon, redshift) {
        _classCallCheck(this, Galaxy);

        this.setPlateIfu(plateifu);
        this.toggleon = toggleon;
        this.redshift = redshift;
        // main elements
        this.maindiv = $('#' + this.plateifu);
        this.metadiv = this.maindiv.find('#metadata');
        this.specdiv = this.maindiv.find('#specview');
        this.imagediv = this.specdiv.find('#imagediv');
        this.mapsdiv = this.specdiv.find('#mapsdiv');
        this.mapdiv = this.specdiv.find('#mapdiv1');
        this.graphdiv = this.specdiv.find('#graphdiv');
        this.specmsg = this.specdiv.find('#specmsg');
        this.mapmsg = this.specdiv.find('#mapmsg');
        this.webspec = null;
        this.staticdiv = this.specdiv.find('#staticdiv');
        this.dynamicdiv = this.specdiv.find('#dynamicdiv');
        this.maptab = $('#maptab');
        // toggle elements
        this.togglediv = $('#toggleinteract');
        this.toggleload = $('#toggle-load');
        this.togglediv.bootstrapToggle('off');
        this.toggleframe = $('#toggleframe');
        this.togglelines = $('#togglelines');
        this.togglemask = $('#togglemask');
        // flag popover elements
        this.qualpop = $('#qualitypopover');
        this.mapqualpop = $('#mapqualitypopover');
        this.targpops = $('.targpopovers');
        // maps elements
        this.dapmapsbut = $('#dapmapsbut');
        this.dapselect = $('#dapmapchoices');
        this.dapbt = $('#dapbtchoices');
        //this.dapselect.selectpicker('deselectAll');
        this.resetmapsbut = $('#resetmapsbut');
        // nsa elements
        this.nsadisplay = $('#nsadisp'); // the NSA Display tab element
        this.nsaplots = $('.marvinplot'); // list of divs for the NSA highcharts scatter plot
        this.nsaplotdiv = this.maindiv.find('#nsahighchart1'); // the first div - NSA scatter plot
        this.nsaboxdiv = this.maindiv.find('#nsad3box'); // the NSA D3 boxplot element
        this.nsaselect = $('.nsaselect'); //$('#nsachoices1');   // list of the NSA selectpicker elements
        this.nsamsg = this.maindiv.find('#nsamsg'); // the NSA error message element
        this.nsaresetbut = $('.nsareset'); //$('#resetnsa1');    // list of the NSA reset button elements
        this.nsamovers = $('#nsatable').find('.mover'); // list of all NSA table parameter name elements
        this.nsaplotbuttons = $('.nsaplotbuts'); // list of the NSA plot button elements
        this.nsatable = $('#nsatable'); // the NSA table element
        this.nsaload = $('#nsa-load'); //the NSA scatter plot loading element

        // object for mapping magnitude bands to their array index
        this.magband = { 'F': 0, 'N': 1, 'u': 2, 'g': 3, 'r': 4, 'i': 5, 'z': 6 };

        // init some stuff
        this.initFlagPopovers();

        //Event Handlers
        this.maptab.on('click', this, this.resizeSpecView); // this event fires when a user clicks the MapSpec View Tab
        this.dapmapsbut.on('click', this, this.getDapMaps); // this event fires when a user clicks the GetMaps button
        this.resetmapsbut.on('click', this, this.resetMaps); // this event fires when a user clicks the Maps Reset button
        this.togglediv.on('change', this, this.initDynamic); // this event fires when a user clicks the Spec/Map View Toggle
        this.nsadisplay.on('click', this, this.displayNSA); // this event fires when a user clicks the NSA tab
        this.nsaresetbut.on('click', this, this.resetNSASelect); // this event fires when a user clicks the NSA select reset button
        //this.nsaselect.on('changed.bs.select', this, this.updateNSAPlot); // this event fires when a user selects an NSA parameter
        this.nsaplotbuttons.on('click', this, this.updateNSAPlot);
        //this.nsatable.on('page-change.bs.table', this, this.updateTableEvents);
        //this.nsatable.on('page-change.bs.table', this, this.updateTableEvents);
        this.toggleframe.on('change', this, this.toggleWavelength); // this event fires when a user clicks the Toggle Obs/Rest Frame
        this.togglelines.on('change', this, this.toggleLines); // this event fires when a user clicks the Toggle Lines
        this.togglemask.on('change', this, this.toggleMask); // this event fires when a user clicks the Toggle Masks
    }

    // Test print


    _createClass(Galaxy, [{
        key: 'print',
        value: function print() {
            console.log('We are now printing galaxy', this.plateifu, this.plate, this.ifu);
        }

        // Set the plateifu

    }, {
        key: 'setPlateIfu',
        value: function setPlateIfu(plateifu) {
            if (plateifu === undefined) {
                this.plateifu = $('.singlegalaxy').attr('id');
            } else {
                this.plateifu = plateifu;
            }

            var _plateifu$split = this.plateifu.split('-');

            var _plateifu$split2 = _slicedToArray(_plateifu$split, 2);

            this.plate = _plateifu$split2[0];
            this.ifu = _plateifu$split2[1];
        }

        // Resize the ouput MapSpec View when tab clicked

    }, {
        key: 'resizeSpecView',
        value: function resizeSpecView(event) {
            var _this = event.data;
            // wait 10 milliseconds before resizing so divs will have the correct size
            m.utils.window[0].setTimeout(function () {
                _this.webspec.resize();
                _this.olmap.map.updateSize();
            }, 10);
        }

        // Compute rest-frame wavelength

    }, {
        key: 'computeRestWave',
        value: function computeRestWave() {
            var _this4 = this;

            var rest = this.setSpectrumAxisFormatter('rest', this.redshift);
            this.obswave = this._spaxeldata.map(function (x) {
                return x[0];
            });
            this.restwave = this.obswave.map(function (x) {
                return rest(x);
            });
            this.rest_spaxeldata = this._spaxeldata.map(function (x, i) {
                return [_this4.restwave[i], x.slice(1)[0], x.slice(1)[1]];
            });
        }

        // Initialize and Load a DyGraph spectrum

    }, {
        key: 'loadSpaxel',
        value: function loadSpaxel(spaxel, title) {
            // this plugin renables dygraphs 1.1 behaviour of unzooming to specified valueRange 
            var doubleClickZoomOutPlugin = {
                activate: function activate(g) {
                    // Save the initial y-axis range for later.
                    var initialValueRange = g.getOption('valueRange');
                    return {
                        dblclick: function dblclick(e) {
                            e.dygraph.updateOptions({
                                dateWindow: null, // zoom all the way out
                                valueRange: initialValueRange // zoom to a specific y-axis range.
                            });
                            e.preventDefault(); // prevent the default zoom out action.
                        }
                    };
                }
            };

            // run intial setup of spaxel data
            this.setupSpaxel(spaxel);

            // initialize Dygraph
            var labels = spaxel[0].length == 3 ? ['Wavelength', 'Flux', 'Model Fit'] : ['Wavelength', 'Flux'];
            var options = {
                title: title,
                labels: labels,
                legend: 'always',
                errorBars: true, // TODO DyGraph shows 2-sigma error bars FIX THIS
                ylabel: 'Flux [10<sup>-17</sup> erg/cm<sup>2</sup>/s/Å]',
                valueRange: [0, null],
                plugins: [doubleClickZoomOutPlugin],
                underlayCallback: this.underlay
            };
            // set up the wavelength axis
            var data = this.toggleframe.prop('checked') ? this.rest_spaxeldata : spaxel;
            options = this.addDygraphWaveOptions(options);

            // create the Dygraph object
            this.webspec = new Dygraph(this.graphdiv[0], data, options);

            // assign the wavelength data and bad mask regions to the Dygraph object; makes accessibile inside underlay callback
            this.webspec.wave = this.toggleframe.prop('checked') ? this.restwave : this.obswave;
            this.webspec.badspots = this.badspots;

            // create dap line annotations and conditionally display
            this.updateAnnotations();
        }

        // This function draws highlighted masks on the Dygraph canvas

    }, {
        key: 'underlay',
        value: function underlay(canvas, area, g) {

            canvas.fillStyle = 'rgb(205, 92, 92, 0.5)';

            if (g.badspots == null) {
                return;
            }

            function highlight_mask(x_start, x_end) {
                var bottom_left = g.toDomCoords(x_start, -20);
                var top_right = g.toDomCoords(x_end, +20);

                var left = bottom_left[0];
                var right = top_right[0];
                canvas.fillRect(left, area.y, right - left, area.h);
            }
            g.badspots.forEach(function (x) {
                var xstart = g.wave[x[0]];
                var xend = g.wave[x[1]];
                highlight_mask(xstart, xend);
            });
        }

        // set up some spaxel data arrays

    }, {
        key: 'setupSpaxel',
        value: function setupSpaxel(spaxel) {
            this._spaxeldata = spaxel;
            this.computeRestWave();
            this.annotations = this.daplines.map(this.createAnnotation, this);
        }

        // Dygraph Axis Formatter

    }, {
        key: 'setSpectrumAxisFormatter',
        value: function setSpectrumAxisFormatter(wave, redshift) {
            var obs = function obs(d, gran) {
                return d;
            };
            var rest = function rest(d, gran) {
                return parseFloat((d / (1 + redshift)).toPrecision(5));
            };

            if (wave === 'obs') {
                return obs;
            } else if (wave === 'rest') {
                return rest;
            }
        }

        // update the Dygraph x-axis label and data with rest-frame / obs wavelength

    }, {
        key: 'addDygraphWaveOptions',
        value: function addDygraphWaveOptions(oldoptions) {
            var newopts = {};
            if (this.toggleframe.prop('checked')) {
                newopts = { 'file': this.rest_spaxeldata, 'xlabel': 'Rest Wavelength [Ångströms]' };
            } else {
                newopts = { 'file': this._spaxeldata, 'xlabel': 'Observed Wavelength [Ångströms]' };
            }
            var options = Object.assign(oldoptions, newopts);
            return options;
        }

        // Toggle the Observed/Rest Wavelength

    }, {
        key: 'toggleWavelength',
        value: function toggleWavelength(event) {
            var _this = event.data;
            var options = {};
            options = _this.addDygraphWaveOptions(options);
            _this.webspec.updateOptions(options);
            _this.updateAnnotations();
            _this.updateMask(_this.badspots);
        }

        // Update the line annotations on Dygraph

    }, {
        key: 'updateAnnotations',
        value: function updateAnnotations() {
            if (this.togglelines.prop('checked')) {
                this.annotations = this.daplines.map(this.createAnnotation, this);
                this.webspec.setAnnotations(this.annotations);
            } else {
                this.webspec.setAnnotations([]);
            }
        }

        // Toggle Line Display

    }, {
        key: 'toggleLines',
        value: function toggleLines(event) {
            var _this = event.data;
            _this.updateAnnotations();
        }

        // create a Dygraph annotation for measured DAP lines

    }, {
        key: 'createAnnotation',
        value: function createAnnotation(x, index, arr) {
            var _x$split = x.split(' '),
                _x$split2 = _slicedToArray(_x$split, 2),
                name = _x$split2[0],
                wave = _x$split2[1];

            name = name.includes('-') ? name.split('-')[0] + name.split('-')[1][0] : name;
            var owave = parseInt(wave) * (1 + this.redshift);
            var wavedata = this.toggleframe.prop('checked') ? this.restwave : this.obswave;
            var use_wave = this.toggleframe.prop('checked') ? wave : owave;
            var diff = wavedata.map(function (y) {
                return Math.abs(y - use_wave);
            });
            var idx = diff.indexOf(Math.min.apply(Math, _toConsumableArray(diff)));
            var closest_wave = String(wavedata[idx]);
            var annot = { 'series': "Flux", 'x': closest_wave, 'shortText': name, 'text': x, 'width': 30, 'height': 20, 'tickHeight': 40, 'tickColor': '#cd5c5c', 'cssClass': 'annotation' };
            return annot;
        }

        // Toggle DONOTUSE Masks

    }, {
        key: 'toggleMask',
        value: function toggleMask(event) {
            var _this = event.data;
            if (_this.togglemask.prop('checked')) {
                _this.webspec.badspots = null;
            } else {
                _this.webspec.badspots = _this.badspots;
            }
            _this.webspec.updateOptions({});
        }

        // Update the masks with new bad regions 

    }, {
        key: 'updateMask',
        value: function updateMask(badspots) {
            this.badspots = badspots == null ? this.badspots : badspots;
            this.webspec.wave = this.toggleframe.prop('checked') ? this.restwave : this.obswave;
            this.webspec.badspots = this.togglemask.prop('checked') ? null : badspots;
            this.webspec.updateOptions({});
        }

        // Update the spectrum message div for errors only

    }, {
        key: 'updateSpecMsg',
        value: function updateSpecMsg(specmsg, status) {
            this.specmsg.hide();
            if (status !== undefined && status === -1) {
                this.specmsg.show();
            }
            specmsg = specmsg.replace('<', '').replace('>', '');
            var newmsg = '<strong>' + specmsg + '</strong>';
            this.specmsg.empty();
            this.specmsg.html(newmsg);
        }

        // Update a DyGraph spectrum after clicking on a Map or the Image

    }, {
        key: 'updateSpaxel',
        value: function updateSpaxel(spaxel, specmsg, badspots) {
            this.setupSpaxel(spaxel);
            this.updateSpecMsg(specmsg);
            var options = { 'title': specmsg };
            options = this.addDygraphWaveOptions(options);
            this.webspec.updateOptions(options);
            this.updateAnnotations();
            this.updateMask(badspots);
        }

        // Initialize OpenLayers Map

    }, {
        key: 'initOpenLayers',
        value: function initOpenLayers(image) {
            this.image = image;
            this.olmap = new OLMap(image);
            // add click event handler on map to get spaxel
            this.olmap.map.on('singleclick', this.getSpaxel, this);
        }

        // Initialize the DAP Heatmap displays

    }, {
        key: 'initHeatmap',
        value: function initHeatmap(maps) {
            var mapchildren = this.mapsdiv.children('div');
            var _this = this;
            $.each(mapchildren, function (index, child) {
                var mapdiv = $(child).find('div').first();
                mapdiv.empty();
                if (maps[index] !== undefined && maps[index].data !== null) {
                    this.heatmap = new HeatMap(mapdiv, maps[index].data, maps[index].msg, maps[index].plotparams, _this);
                    this.heatmap.mapdiv.highcharts().reflow();
                }
            });
        }

        // Make Promise error message

    }, {
        key: 'makeError',
        value: function makeError(name) {
            return 'Unknown Error: the ' + name + ' javascript Ajax request failed!';
        }

        // Retrieves a new Spaxel from the server based on a given mouse position or xy spaxel coord.

    }, {
        key: 'getSpaxel',
        value: function getSpaxel(event) {
            var _this5 = this;

            var mousecoords = event.coordinate === undefined ? null : event.coordinate;
            var divid = $(event.target).parents('div').first().attr('id');
            var maptype = divid !== undefined && divid.search('highcharts') !== -1 ? 'heatmap' : 'optical';
            var x = event.point === undefined ? null : event.point.x;
            var y = event.point === undefined ? null : event.point.y;
            var keys = ['plateifu', 'image', 'imwidth', 'imheight', 'mousecoords', 'type', 'x', 'y'];
            var form = m.utils.buildForm(keys, this.plateifu, this.image, this.olmap.imwidth, this.olmap.imheight, mousecoords, maptype, x, y);

            // send the form data
            Promise.resolve($.post(Flask.url_for('galaxy_page.getspaxel'), form, 'json')).then(function (data) {
                if (data.result.status === -1) {
                    throw new SpaxelError('Error: ' + data.result.specmsg);
                }
                _this5.updateSpaxel(data.result.spectra, data.result.specmsg, data.result.badspots);
            }).catch(function (error) {
                var errmsg = error.message === undefined ? _this5.makeError('getSpaxel') : error.message;
                _this5.updateSpecMsg(errmsg, -1);
            });
        }

        // check the toggle preference on initial page load
        // eventually for user preferences

    }, {
        key: 'checkToggle',
        value: function checkToggle() {
            if (this.toggleon === 'true') {
                this.toggleOn();
            } else {
                this.toggleOff();
            }
        }

        // toggle the display button on

    }, {
        key: 'toggleOn',
        value: function toggleOn() {
            // eventually this should include the ajax stuff inside initDynamic - for after user preferences implemented
            this.toggleon = true;
            //this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
            //this.togglediv.button('complete');
            this.staticdiv.hide();
            this.dynamicdiv.show();
        }

        // toggle the display button off

    }, {
        key: 'toggleOff',
        value: function toggleOff() {
            this.toggleon = false;
            //this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
            //this.togglediv.button('reset');
            this.dynamicdiv.hide();
            this.staticdiv.show();
        }
    }, {
        key: 'testTogg',
        value: function testTogg(event) {
            var _this = event.data;
            console.log('toggling', _this.togglediv.prop('checked'), _this.togglediv.hasClass('active'));
        }

        // Initialize the Dynamic Galaxy Interaction upon toggle - makes loading an AJAX request

    }, {
        key: 'initDynamic',
        value: function initDynamic(event) {
            var _this6 = this;

            var _this = event.data;

            if (!_this.togglediv.prop('checked')) {
                // Turning Off
                _this.toggleOff();
            } else {
                // Turning On
                _this.toggleOn();

                // check for empty divs
                var specempty = _this.graphdiv.is(':empty');
                var imageempty = _this.imagediv.is(':empty');
                var mapempty = _this.mapdiv.is(':empty');

                // send the request if the dynamic divs are empty
                if (imageempty) {
                    // make the form
                    var keys = ['plateifu', 'toggleon'];
                    var form = m.utils.buildForm(keys, _this.plateifu, _this.toggleon);
                    _this.toggleload.show();

                    // send the form data
                    Promise.resolve($.post(Flask.url_for('galaxy_page.initdynamic'), form, 'json')).then(function (data) {
                        if (data.result.error) {
                            var err = data.result.error;
                            throw new SpaxelError('Error : ' + err);
                        }
                        if (data.result.specstatus === -1) {
                            throw new SpaxelError('Error: ' + data.result.specmsg);
                        }
                        if (data.result.mapstatus === -1) {
                            throw new MapError('Error: ' + data.result.mapmsg);
                        }

                        // extract some properties from the result
                        var image = data.result.image;
                        var spaxel = data.result.spectra;
                        var spectitle = data.result.specmsg;
                        var maps = data.result.maps;
                        var mapmsg = data.result.mapmsg;

                        // add the list of DAP lines and bad mask regions to the global scope
                        _this.daplines = data.result.daplines;
                        _this.badspots = data.result.badspots;

                        // Load the Galaxy Image
                        _this.initOpenLayers(image);
                        _this.toggleload.hide();

                        // Load the Spaxel and Maps
                        _this.loadSpaxel(spaxel, spectitle);
                        _this.initHeatmap(maps);
                        // refresh the map selectpicker
                        _this.dapselect.selectpicker('refresh');
                    }).catch(function (error) {
                        var errmsg = error.message === undefined ? _this6.makeError('initDynamic') : error.message;
                        _this.updateSpecMsg(errmsg, -1);
                        _this.updateMapMsg(errmsg, -1);
                    });
                }
            }
        }

        //  Initialize the Quality and Target Popovers

    }, {
        key: 'initFlagPopovers',
        value: function initFlagPopovers() {
            // DRP Quality Popovers
            this.qualpop.popover({ html: true, content: $('#list_drp3quality').html() });
            // DAP Map Quality Popovers
            this.mapqualpop.popover({ html: true, content: $('#list_dapquality').html() });
            // MaNGA Target Popovers
            $.each(this.targpops, function (index, value) {
                // get id of flag link
                var popid = value.id;
                // split id and grab the mngtarg

                var _popid$split = popid.split('_'),
                    _popid$split2 = _slicedToArray(_popid$split, 2),
                    base = _popid$split2[0],
                    targ = _popid$split2[1];
                // build the label list id


                var listid = '#list_' + targ;
                // init the specific popover
                $('#' + popid).popover({ html: true, content: $(listid).html() });
            });
        }

        // Get some DAP Maps

    }, {
        key: 'getDapMaps',
        value: function getDapMaps(event) {
            var _this = event.data;
            var params = _this.dapselect.selectpicker('val');
            var bintemp = _this.dapbt.selectpicker('val');
            var keys = ['plateifu', 'params', 'bintemp'];
            var form = m.utils.buildForm(keys, _this.plateifu, params, bintemp);
            _this.mapmsg.hide();
            $(this).button('loading');

            // send the form data
            Promise.resolve($.post(Flask.url_for('galaxy_page.updatemaps'), form, 'json')).then(function (data) {
                if (data.result.status === -1) {
                    throw new MapError('Error: ' + data.result.mapmsg);
                }
                _this.dapmapsbut.button('reset');
                _this.initHeatmap(data.result.maps);
            }).catch(function (error) {
                var errmsg = error.message === undefined ? _this.makeError('getDapMaps') : error.message;
                _this.updateMapMsg(errmsg, -1);
                _this.dapmapsbut.button('reset');
            });
        }

        // Update the Map Msg

    }, {
        key: 'updateMapMsg',
        value: function updateMapMsg(mapmsg, status) {
            this.mapmsg.hide();
            if (status !== undefined && status === -1) {
                this.mapmsg.show();
            }
            mapmsg = mapmsg.replace('<', '').replace('>', '');
            var newmsg = '<strong>' + mapmsg + '</strong>';
            this.mapmsg.empty();
            this.mapmsg.html(newmsg);
        }

        // Reset the Maps selection

    }, {
        key: 'resetMaps',
        value: function resetMaps(event) {
            var _this = event.data;
            _this.mapmsg.hide();
            _this.dapselect.selectpicker('deselectAll');
            _this.dapselect.selectpicker('refresh');
        }

        // Set if the galaxy has NSA data or not

    }, {
        key: 'hasNSA',
        value: function hasNSA(hasnsa) {
            this.hasnsa = hasnsa;
        }

        // Display the NSA info

    }, {
        key: 'displayNSA',
        value: function displayNSA(event) {
            var _this = event.data;

            // make the form
            var keys = ['plateifu'];
            var form = m.utils.buildForm(keys, _this.plateifu);

            // send the request if the div is empty
            var nsaempty = _this.nsaplots.is(':empty');
            if (nsaempty & _this.hasnsa) {
                // send the form data
                Promise.resolve($.post(Flask.url_for('galaxy_page.initnsaplot'), form, 'json')).then(function (data) {
                    if (data.result.status !== 1) {
                        throw new Error('Error: ' + data.result.nsamsg);
                    }
                    _this.addNSAData(data.result.nsa);
                    _this.refreshNSASelect(data.result.nsachoices);
                    _this.initNSAScatter();
                    _this.setTableEvents();
                    _this.addNSAEvents();
                    _this.initNSABoxPlot(data.result.nsaplotcols);
                    _this.nsaload.hide();
                }).catch(function (error) {
                    var errmsg = error.message === undefined ? _this.makeError('displayNSA') : error.message;
                    _this.updateNSAMsg(errmsg, -1);
                });
            }
        }

        // add the NSA data into the Galaxy object

    }, {
        key: 'addNSAData',
        value: function addNSAData(data) {
            // the galaxy
            if (data[this.plateifu]) {
                this.mygalaxy = data[this.plateifu];
            } else {
                this.updateNSAMsg('Error: No NSA data found for ' + this.plateifu, -1);
                return;
            }
            // the manga sample
            if (data.sample) {
                this.nsasample = data.sample;
            } else {
                this.updateNSAMsg('Error: Problem getting NSA data found for the MaNGA sample', -1);
                return;
            }
        }

        // get new NSA data based on drag-drop axis change

    }, {
        key: 'updateNSAData',
        value: function updateNSAData(index, type) {
            var _this7 = this;

            var data = void 0,
                options = void 0;
            if (type === 'galaxy') {
                var x = this.mygalaxy[this.nsachoices[index].x];
                var y = this.mygalaxy[this.nsachoices[index].y];
                var pattern = 'absmag_[a-z]$';
                var xrev = this.nsachoices[index].x.search(pattern) > -1 ? true : false;
                var yrev = this.nsachoices[index].y.search(pattern) > -1 ? true : false;
                data = [{ 'name': this.plateifu, 'x': x, 'y': y }];
                options = { xtitle: this.nsachoices[index].xtitle, ytitle: this.nsachoices[index].ytitle,
                    title: this.nsachoices[index].title, galaxy: { name: this.plateifu }, xrev: xrev,
                    yrev: yrev };
            } else if (type === 'sample') {
                var _x = this.nsasample[this.nsachoices[index].x];
                var _y = this.nsasample[this.nsachoices[index].y];
                data = [];
                $.each(_x, function (index, value) {
                    if (value > -9999 && _y[index] > -9999) {
                        var tmp = { 'name': _this7.nsasample.plateifu[index], 'x': value, 'y': _y[index] };
                        data.push(tmp);
                    }
                });
                options = { xtitle: this.nsachoices[index].xtitle, ytitle: this.nsachoices[index].ytitle,
                    title: this.nsachoices[index].title, altseries: { name: 'Sample' } };
            }
            return [data, options];
        }

        // Update the Table event handlers when the table state changes

    }, {
        key: 'setTableEvents',
        value: function setTableEvents() {
            var _this8 = this;

            var tabledata = this.nsatable.bootstrapTable('getData');

            $.each(this.nsamovers, function (index, mover) {
                var id = mover.id;
                $('#' + id).on('dragstart', _this8, _this8.dragStart);
                $('#' + id).on('dragover', _this8, _this8.dragOver);
                $('#' + id).on('drop', _this8, _this8.moverDrop);
            });

            this.nsatable.on('page-change.bs.table', function () {
                $.each(tabledata, function (index, row) {
                    var mover = row[0];
                    var id = $(mover).attr('id');
                    $('#' + id).on('dragstart', _this8, _this8.dragStart);
                    $('#' + id).on('dragover', _this8, _this8.dragOver);
                    $('#' + id).on('drop', _this8, _this8.moverDrop);
                });
            });
        }

        // Add event handlers to the Highcharts scatter plots

    }, {
        key: 'addNSAEvents',
        value: function addNSAEvents() {
            var _this9 = this;

            //let _this = this;
            // NSA plot events
            this.nsaplots = $('.marvinplot');
            $.each(this.nsaplots, function (index, plot) {
                var id = plot.id;
                var highx = $('#' + id).find('.highcharts-xaxis');
                var highy = $('#' + id).find('.highcharts-yaxis');

                highx.on('dragover', _this9, _this9.dragOver);
                highx.on('dragenter', _this9, _this9.dragEnter);
                highx.on('drop', _this9, _this9.dropElement);
                highy.on('dragover', _this9, _this9.dragOver);
                highy.on('dragenter', _this9, _this9.dragEnter);
                highy.on('drop', _this9, _this9.dropElement);
            });
        }

        // Update the NSA Msg

    }, {
        key: 'updateNSAMsg',
        value: function updateNSAMsg(nsamsg, status) {
            this.nsamsg.hide();
            if (status !== undefined && status === -1) {
                this.nsamsg.show();
            }
            var newmsg = '<strong>' + nsamsg + '</strong>';
            this.nsamsg.empty();
            this.nsamsg.html(newmsg);
        }

        // remove values of -9999 from arrays

    }, {
        key: 'filterArray',
        value: function filterArray(value) {
            return value !== -9999.0;
        }

        // create the d3 data format

    }, {
        key: 'createD3data',
        value: function createD3data() {
            var _this10 = this;

            var data = [];
            this.nsaplotcols.forEach(function (column, index) {
                var goodsample = _this10.nsasample[column].filter(_this10.filterArray);
                var tmp = { 'value': _this10.mygalaxy[column], 'title': column, 'sample': goodsample };
                data.push(tmp);
            });
            return data;
        }

        // initialize the NSA d3 box and whisker plot

    }, {
        key: 'initNSABoxPlot',
        value: function initNSABoxPlot(cols) {
            // test for undefined columns
            if (cols === undefined && this.nsaplotcols === undefined) {
                console.error('columns for NSA boxplot are undefined');
            } else {
                this.nsaplotcols = cols;
            }

            // generate the data format
            var data = void 0,
                options = void 0;
            data = this.createD3data();
            this.nsad3box = new BoxWhisker(this.nsaboxdiv, data, options);
        }

        // Destroy old Charts

    }, {
        key: 'destroyChart',
        value: function destroyChart(div, index) {
            this.nsascatter[index].chart.destroy();
            div.empty();
        }

        // Init the NSA Scatter plot

    }, {
        key: 'initNSAScatter',
        value: function initNSAScatter(parentid) {
            var _this11 = this;

            // only update the single parent div element
            if (parentid !== undefined) {
                var parentdiv = this.maindiv.find('#' + parentid);
                var index = parseInt(parentid[parentid.length - 1]);

                var _updateNSAData = this.updateNSAData(index, 'galaxy'),
                    _updateNSAData2 = _slicedToArray(_updateNSAData, 2),
                    data = _updateNSAData2[0],
                    options = _updateNSAData2[1];

                var _updateNSAData3 = this.updateNSAData(index, 'sample'),
                    _updateNSAData4 = _slicedToArray(_updateNSAData3, 2),
                    sdata = _updateNSAData4[0],
                    soptions = _updateNSAData4[1];

                options.altseries = { data: sdata, name: 'Sample' };
                this.destroyChart(parentdiv, index);
                this.nsascatter[index] = new Scatter(parentdiv, data, options);
            } else {
                // try updating all of them
                this.nsascatter = {};
                $.each(this.nsaplots, function (index, plot) {
                    var plotdiv = $(plot);

                    var _updateNSAData5 = _this11.updateNSAData(index + 1, 'galaxy'),
                        _updateNSAData6 = _slicedToArray(_updateNSAData5, 2),
                        data = _updateNSAData6[0],
                        options = _updateNSAData6[1];

                    var _updateNSAData7 = _this11.updateNSAData(index + 1, 'sample'),
                        _updateNSAData8 = _slicedToArray(_updateNSAData7, 2),
                        sdata = _updateNSAData8[0],
                        soptions = _updateNSAData8[1];

                    options.altseries = { data: sdata, name: 'Sample' };
                    _this11.nsascatter[index + 1] = new Scatter(plotdiv, data, options);
                });
            }
        }

        // Refresh the NSA select choices for the scatter plot

    }, {
        key: 'refreshNSASelect',
        value: function refreshNSASelect(vals) {
            this.nsachoices = vals;
            $.each(this.nsaselect, function (index, nsasp) {
                $(nsasp).selectpicker('deselectAll');
                $(nsasp).selectpicker('val', ['x_' + vals[index + 1].x, 'y_' + vals[index + 1].y]);
                $(nsasp).selectpicker('refresh');
            });
        }

        // Update the NSA selectpicker choices for the scatter plot

    }, {
        key: 'updateNSAChoices',
        value: function updateNSAChoices(index, params) {
            var xpar = params[0].slice(2, params[0].length);
            var ypar = params[1].slice(2, params[1].length);
            this.nsachoices[index].title = ypar + ' vs ' + xpar;
            this.nsachoices[index].xtitle = xpar;
            this.nsachoices[index].x = xpar;
            this.nsachoices[index].ytitle = ypar;
            this.nsachoices[index].y = ypar;
        }

        // Reset the NSA selecpicker

    }, {
        key: 'resetNSASelect',
        value: function resetNSASelect(event) {
            var resetid = $(this).attr('id');
            var index = parseInt(resetid[resetid.length - 1]);
            var _this = event.data;
            var myselect = _this.nsaselect[index - 1];
            _this.nsamsg.hide();
            $(myselect).selectpicker('deselectAll');
            $(myselect).selectpicker('refresh');
        }

        // Update the NSA scatter plot on select change

    }, {
        key: 'updateNSAPlot',
        value: function updateNSAPlot(event) {
            var _this = event.data;
            var plotid = $(this).attr('id');
            var index = parseInt(plotid[plotid.length - 1]);
            var nsasp = _this.nsaselect[index - 1];
            var params = $(nsasp).selectpicker('val');

            // Construct the new NSA data
            var parentid = 'nsahighchart' + index;
            _this.updateNSAChoices(index, params);
            _this.initNSAScatter(parentid);
            _this.addNSAEvents();
        }

        // Events for Drag and Drop

        // Element drag start

    }, {
        key: 'dragStart',
        value: function dragStart(event) {
            var _this = event.data;
            var param = this.id + '+' + this.textContent;
            event.originalEvent.dataTransfer.setData('Text', param);

            // show the overlay elements
            $.each(_this.nsascatter, function (index, scat) {
                scat.overgroup.show();
            });
        }
        // Element drag over

    }, {
        key: 'dragOver',
        value: function dragOver(event) {
            event.preventDefault();
            //event.stopPropagation();
            event.originalEvent.dataTransfer.dropEffect = 'move';
        }
        // Element drag enter

    }, {
        key: 'dragEnter',
        value: function dragEnter(event) {
            event.preventDefault();
            //event.stopPropagation();
        }
        // Mover element drop event

    }, {
        key: 'moverDrop',
        value: function moverDrop(event) {
            event.preventDefault();
            event.stopPropagation();
        }
        // Element drop and redraw the scatter plot

    }, {
        key: 'dropElement',
        value: function dropElement(event) {
            event.preventDefault();
            event.stopPropagation();
            // get the id and name of the dropped parameter
            var _this = event.data;
            var param = event.originalEvent.dataTransfer.getData('Text');

            var _param$split = param.split('+'),
                _param$split2 = _slicedToArray(_param$split, 2),
                id = _param$split2[0],
                name = _param$split2[1];

            // Hide overlay elements


            $.each(_this.nsascatter, function (index, scat) {
                scat.overgroup.hide();
            });

            // Determine which axis and plot the name was dropped on
            var classes = $(this).attr('class');
            var isX = classes.includes('highcharts-xaxis');
            var isY = classes.includes('highcharts-yaxis');
            var parentdiv = $(this).closest('.marvinplot');
            var parentid = parentdiv.attr('id');
            if (parentid === undefined) {
                event.stopPropagation();
                return false;
            }
            var parentindex = parseInt(parentid[parentid.length - 1]);

            // get the other axis and extract title
            var otheraxis = null;
            if (isX) {
                otheraxis = $(this).next();
            } else if (isY) {
                otheraxis = $(this).prev();
            }
            var axistitle = this.textContent;
            var otheraxistitle = otheraxis[0].textContent;

            // Update the Values
            var newtitle = _this.nsachoices[parentindex].title.replace(axistitle, name);
            _this.nsachoices[parentindex].title = newtitle;
            if (isX) {
                _this.nsachoices[parentindex].xtitle = name;
                _this.nsachoices[parentindex].x = id;
            } else if (isY) {
                _this.nsachoices[parentindex].ytitle = name;
                _this.nsachoices[parentindex].y = id;
            }

            // Construct the new NSA data
            _this.initNSAScatter(parentid);
            _this.addNSAEvents();

            return false;
        }
    }]);

    return Galaxy;
}();
