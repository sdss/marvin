/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 16:49:00
* @Last Modified by:   Brian Cherinka
<<<<<<< HEAD
* @Last Modified time: 2017-02-21 16:26:48
=======
* @Last Modified time: 2016-09-26 17:40:15
>>>>>>> upstream/marvin_refactor
*/

//
// Javascript Galaxy object handling JS things for a single galaxy
//

'use strict';

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Galaxy = function () {

    // Constructor
    function Galaxy(plateifu, toggleon) {
        _classCallCheck(this, Galaxy);

        this.setPlateIfu(plateifu);
        this.toggleon = toggleon;
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
        // toggle elements
        this.togglediv = $('#toggleinteract');
        this.toggleload = $('#toggle-load');
        this.togglediv.bootstrapToggle('off');
        // flag popover elements
        this.qualpop = $('#qualitypopover');
        this.targpops = $('.targpopovers');
        // maps elements
        this.dapmapsbut = $('#dapmapsbut');
        this.dapselect = $('#dapmapchoices');
        this.dapbt = $('#dapbtchoices');
        this.dapselect.selectpicker('deselectAll');
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
        //this.checkToggle();

        //Event Handlers
        this.dapmapsbut.on('click', this, this.getDapMaps); // this event fires when a user clicks the GetMaps button
        this.resetmapsbut.on('click', this, this.resetMaps); // this event fires when a user clicks the Maps Reset button
        this.togglediv.on('change', this, this.initDynamic); // this event fires when a user clicks the Spec/Map View Toggle
        this.nsadisplay.on('click', this, this.displayNSA); // this event fires when a user clicks the NSA tab
        this.nsaresetbut.on('click', this, this.resetNSASelect); // this event fires when a user clicks the NSA select reset button
        //this.nsaselect.on('changed.bs.select', this, this.updateNSAPlot); // this event fires when a user selects an NSA parameter
        this.nsaplotbuttons.on('click', this, this.updateNSAPlot);
        //this.nsatable.on('page-change.bs.table', this, this.updateTableEvents);
        //this.nsatable.on('page-change.bs.table', this, this.updateTableEvents);

        // NSA movers events
        // var _this = this;
        // $.each(this.nsamovers, function(index, mover) {
        //     var id = mover.id;
        //     $('#'+id).on('dragstart', this, _this.dragStart);
        //     $('#'+id).on('dragover', this, _this.dragOver);
        // });
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

        // Initialize and Load a DyGraph spectrum

    }, {
        key: 'loadSpaxel',
        value: function loadSpaxel(spaxel, title) {
            var labels = spaxel[0].length == 3 ? ['Wavelength', 'Flux', 'Model Fit'] : ['Wavelength', 'Flux'];
            this.webspec = new Dygraph(this.graphdiv[0], spaxel, {
                title: title,
                labels: labels,
                errorBars: true, // TODO DyGraph shows 2-sigma error bars FIX THIS
                ylabel: 'Flux [10<sup>-17</sup> erg/cm<sup>2</sup>/s/Å]',
                xlabel: 'Wavelength [Ångströms]'
            });
        }

        // Update the spectrum message div for errors only

    }, {
        key: 'updateSpecMsg',
        value: function updateSpecMsg(specmsg, status) {
            this.specmsg.hide();
            if (status !== undefined && status === -1) {
                this.specmsg.show();
            }
            var newmsg = '<strong>' + specmsg + '</strong>';
            this.specmsg.empty();
            this.specmsg.html(newmsg);
        }

        // Update a DyGraph spectrum

    }, {
        key: 'updateSpaxel',
        value: function updateSpaxel(spaxel, specmsg) {
            this.updateSpecMsg(specmsg);
            this.webspec.updateOptions({ 'file': spaxel, 'title': specmsg });
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
    }, {
        key: 'initHeatmap',
        value: function initHeatmap(maps) {
            console.log('initHeatmap', this.mapsdiv);
            var mapchildren = this.mapsdiv.children('div');
            console.log('mapchildren', mapchildren);
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

        // Retrieves a new Spaxel from the server based on a given mouse position or xy spaxel coord.

    }, {
        key: 'getSpaxel',
        value: function getSpaxel(event) {
            var mousecoords = event.coordinate === undefined ? null : event.coordinate;
            var divid = $(event.target).parents('div').first().attr('id');
            var maptype = divid !== undefined && divid.search('highcharts') !== -1 ? 'heatmap' : 'optical';
            var x = event.point === undefined ? null : event.point.x;
            var y = event.point === undefined ? null : event.point.y;
            var keys = ['plateifu', 'image', 'imwidth', 'imheight', 'mousecoords', 'type', 'x', 'y'];
            var form = m.utils.buildForm(keys, this.plateifu, this.image, this.olmap.imwidth, this.olmap.imheight, mousecoords, maptype, x, y);
            var _this = this;

            // send the form data
            $.post(Flask.url_for('galaxy_page.getspaxel'), form, 'json').done(function (data) {
                if (data.result.status !== -1) {
                    _this.updateSpaxel(data.result.spectra, data.result.specmsg);
                } else {
                    _this.updateSpecMsg('Error: ' + data.result.specmsg, data.result.status);
                }
            }).fail(function (data) {
                _this.updateSpecMsg('Error: ' + data.result.specmsg, data.result.status);
            });
        }

        // check the toggle preference on initial page load
        // eventually for user preferences

    }, {
        key: 'checkToggle',
        value: function checkToggle() {
            if (this.toggleon) {
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

                    $.post(Flask.url_for('galaxy_page.initdynamic'), form, 'json').done(function (data) {

                        var image = data.result.image;
                        var spaxel = data.result.spectra;
                        var spectitle = data.result.specmsg;
                        var maps = data.result.maps;
                        var mapmsg = data.result.mapmsg;

                        // Load the Galaxy Image
                        _this.initOpenLayers(image);
                        _this.toggleload.hide();

                        // Try to load the spaxel
                        if (data.result.specstatus !== -1) {
                            _this.loadSpaxel(spaxel, spectitle);
                        } else {
                            _this.updateSpecMsg('Error: ' + spectitle, data.result.specstatus);
                        }

                        // Try to load the Maps
                        if (data.result.mapstatus !== -1) {
                            _this.initHeatmap(maps);
                        } else {
                            _this.updateMapMsg('Error: ' + mapmsg, data.result.mapstatus);
                        }
                    }).fail(function (data) {
                        _this.updateSpecMsg('Error: ' + data.result.specmsg, data.result.specstatus);
                        _this.updateMapMsg('Error: ' + data.result.mapmsg, data.result.mapstatus);
                        _this.toggleload.hide();
                    });
                }
            }
        }

        // Toggle the interactive OpenLayers map and Dygraph spectra
        // DEPRECATED - REMOVE

    }, {
        key: 'toggleInteract',
        value: function toggleInteract(image, maps, spaxel, spectitle, mapmsg) {
            if (this.togglediv.hasClass('active')) {
                // Turning Off
                this.toggleon = false;
                this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
                this.togglediv.button('reset');
                this.dynamicdiv.hide();
                this.staticdiv.show();
            } else {
                // Turning On
                this.toggleon = true;
                this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
                this.togglediv.button('complete');
                this.staticdiv.hide();
                this.dynamicdiv.show();

                // check for empty divs
                var specempty = this.graphdiv.is(':empty');
                var imageempty = this.imagediv.is(':empty');
                var mapempty = this.mapdiv.is(':empty');
                // load the spaxel if the div is initially empty;
                if (this.graphdiv !== undefined && specempty) {
                    this.loadSpaxel(spaxel, spectitle);
                }
                // load the image if div is empty
                if (imageempty) {
                    this.initOpenLayers(image);
                }
                // load the map if div is empty
                if (mapempty) {
                    this.initHeatmap(maps);
                }

                // possibly update an initial map message
                if (mapmsg !== null) {
                    this.updateMapMsg(mapmsg, -1);
                }
            }
        }

        //  Initialize the Quality and Target Popovers

    }, {
        key: 'initFlagPopovers',
        value: function initFlagPopovers() {
            // DRP Quality Popovers
            this.qualpop.popover({ html: true, content: $('#list_drp3quality').html() });
            // MaNGA Target Popovers
            $.each(this.targpops, function (index, value) {
                // get id of flag link
                var popid = value.id;
                // split id and grab the mngtarg

                var _popid$split = popid.split('_');

                var _popid$split2 = _slicedToArray(_popid$split, 2);

                var base = _popid$split2[0];
                var targ = _popid$split2[1];
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
            $.post(Flask.url_for('galaxy_page.updatemaps'), form, 'json').done(function (data) {
                if (data.result.status !== -1) {
                    _this.dapmapsbut.button('reset');
                    _this.initHeatmap(data.result.maps);
                } else {
                    _this.updateMapMsg('Error: ' + data.result.mapmsg, data.result.status);
                    _this.dapmapsbut.button('reset');
                }
            }).fail(function (data) {
                _this.updateMapMsg('Error: ' + data.result.mapmsg, data.result.status);
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
                $.post(Flask.url_for('galaxy_page.initnsaplot'), form, 'json').done(function (data) {
                    if (data.result.status !== -1) {
                        _this.addNSAData(data.result.nsa);
                        _this.refreshNSASelect(data.result.nsachoices);
                        _this.initNSAScatter();
                        _this.setTableEvents();
                        _this.addNSAEvents();
                        _this.initNSABoxPlot(data.result.nsaplotcols);
                        _this.nsaload.hide();
                    } else {
                        _this.updateNSAMsg('Error: ' + data.result.nsamsg, data.result.status);
                    }
                }).fail(function (data) {
                    _this.updateNSAMsg('Error: ' + data.result.nsamsg, data.result.status);
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
            var data, options;
            var _this = this;
            if (type === 'galaxy') {
                var x = this.mygalaxy[this.nsachoices[index].x];
                var y = this.mygalaxy[this.nsachoices[index].y];
                var xrev = this.nsachoices[index].x.search('absmag') > -1 ? true : false;
                var yrev = this.nsachoices[index].y.search('absmag') > -1 ? true : false;
                data = [{ 'name': this.plateifu, 'x': x, 'y': y }];
                options = { xtitle: this.nsachoices[index].xtitle, ytitle: this.nsachoices[index].ytitle,
                    title: this.nsachoices[index].title, galaxy: { name: this.plateifu }, xrev: xrev,
                    yrev: yrev };
            } else if (type === 'sample') {
                var x = this.nsasample[this.nsachoices[index].x];
                var y = this.nsasample[this.nsachoices[index].y];
                data = [];
                $.each(x, function (index, value) {
                    if (value > -9999 && y[index] > -9999) {
                        var tmp = { 'name': _this.nsasample.plateifu[index], 'x': value, 'y': y[index] };
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
            var tabledata = this.nsatable.bootstrapTable('getData');
            var _this = this;

            $.each(this.nsamovers, function (index, mover) {
                var id = mover.id;
                $('#' + id).on('dragstart', _this, _this.dragStart);
                $('#' + id).on('dragover', _this, _this.dragOver);
                $('#' + id).on('drop', _this, _this.moverDrop);
            });

            this.nsatable.on('page-change.bs.table', function () {
                $.each(tabledata, function (index, row) {
                    var mover = row[0];
                    var id = $(mover).attr('id');
                    $('#' + id).on('dragstart', _this, _this.dragStart);
                    $('#' + id).on('dragover', _this, _this.dragOver);
                    $('#' + id).on('drop', _this, _this.moverDrop);
                });
            });
        }

        // Add event handlers to the Highcharts scatter plots

    }, {
        key: 'addNSAEvents',
        value: function addNSAEvents() {
            var _this = this;
            // NSA plot events
            this.nsaplots = $('.marvinplot');
            $.each(this.nsaplots, function (index, plot) {
                var id = plot.id;
                var highx = $('#' + id).find('.highcharts-xaxis');
                var highy = $('#' + id).find('.highcharts-yaxis');

                highx.on('dragover', _this, _this.dragOver);
                highx.on('dragenter', _this, _this.dragEnter);
                highx.on('drop', _this, _this.dropElement);
                highy.on('dragover', _this, _this.dragOver);
                highy.on('dragenter', _this, _this.dragEnter);
                highy.on('drop', _this, _this.dropElement);
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
            var data = [];
            var _this = this;
            $.each(this.nsaplotcols, function (index, column) {
                var goodsample = _this.nsasample[column].filter(_this.filterArray);
                var tmp = { 'value': _this.mygalaxy[column], 'title': column, 'sample': goodsample };
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
            var data, options;
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
            var _this = this;
            // only update the single parent div element
            if (parentid !== undefined) {
                var parentdiv = this.maindiv.find('#' + parentid);
                var index = parseInt(parentid[parentid.length - 1]);

                var _updateNSAData = this.updateNSAData(index, 'galaxy');

                var _updateNSAData2 = _slicedToArray(_updateNSAData, 2);

                var data = _updateNSAData2[0];
                var options = _updateNSAData2[1];

                var _updateNSAData3 = this.updateNSAData(index, 'sample');

                var _updateNSAData4 = _slicedToArray(_updateNSAData3, 2);

                var sdata = _updateNSAData4[0];
                var soptions = _updateNSAData4[1];

                options['altseries'] = { data: sdata, name: 'Sample' };
                this.destroyChart(parentdiv, index);
                this.nsascatter[index] = new Scatter(parentdiv, data, options);
            } else {
                // try updating all of them
                _this.nsascatter = {};
                $.each(this.nsaplots, function (index, plot) {
                    var plotdiv = $(plot);

                    var _this$updateNSAData = _this.updateNSAData(index + 1, 'galaxy');

                    var _this$updateNSAData2 = _slicedToArray(_this$updateNSAData, 2);

                    var data = _this$updateNSAData2[0];
                    var options = _this$updateNSAData2[1];

                    var _this$updateNSAData3 = _this.updateNSAData(index + 1, 'sample');

                    var _this$updateNSAData4 = _slicedToArray(_this$updateNSAData3, 2);

                    var sdata = _this$updateNSAData4[0];
                    var soptions = _this$updateNSAData4[1];

                    options['altseries'] = { data: sdata, name: 'Sample' };
                    _this.nsascatter[index + 1] = new Scatter(plotdiv, data, options);
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

            var _param$split = param.split('+');

            var _param$split2 = _slicedToArray(_param$split, 2);

            var id = _param$split2[0];
            var name = _param$split2[1];

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
