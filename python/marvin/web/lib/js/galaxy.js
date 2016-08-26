/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 16:49:00
* @Last Modified by:   Brian
* @Last Modified time: 2016-05-23 13:54:39
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
    function Galaxy(plateifu) {
        _classCallCheck(this, Galaxy);

        this.setPlateIfu(plateifu);
        this.maindiv = $('#' + this.plateifu);
        this.metadiv = this.maindiv.find('#metadata');
        this.specdiv = this.maindiv.find('#specview');
        this.imagediv = this.specdiv.find('#imagediv');
        this.mapdiv = this.specdiv.find('#mapdiv');
        this.graphdiv = this.specdiv.find('#graphdiv');
        this.specmsg = this.specdiv.find('#specmsg');
        this.webspec = null;
        this.staticdiv = this.specdiv.find('#staticdiv');
        this.dynamicdiv = this.specdiv.find('#dynamicdiv');
        this.togglediv = $('#toggleinteract');
        this.qualpop = $('#qualitypopover');
        this.targpops = $('.targpopovers');

        // init some stuff
        this.initFlagPopovers();
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
            this.webspec = new Dygraph(this.graphdiv[0], spaxel, {
                title: title,
                labels: ['Wavelength', 'Flux'],
                errorBars: true, // TODO DyGraph shows 2-sigma error bars FIX THIS
                ylabel: 'Flux [10<sup>-17</sup> erg/cm<sup>2</sup>/s/Å]',
                xlabel: 'Wavelength [Ångströms]'
            });
        }
    }, {
        key: 'updateSpecMsg',


        // Update the spectrum message div for errors only
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
    }, {
        key: 'initOpenLayers',


        // Initialize OpenLayers Map
        value: function initOpenLayers(image) {
            this.image = image;
            this.olmap = new OLMap(image);
            // add click event handler on map to get spaxel
            this.olmap.map.on('singleclick', this.getSpaxel, this);
        }
    }, {
        key: 'initHeatmap',


        // Initialize Highcharts heatmap
        value: function initHeatmap(myjson, spaxel, maptitle, spectitle, divname, chartWidth) {
            console.log('initHeatmap chartWidth', chartWidth);
            var _this = this;
            $(function () {
                var $container = $('#' + divname),
                    chart;
                var cubeside = 34;
                $container.highcharts({
                    chart: { type: 'heatmap',
                        marginTop: 40,
                        marginBottom: 80,
                        plotBorderWidth: 1,
                        panning: true,
                        panKey: 'shift',
                        zoomType: 'xy',
                        alignticks: false
                    },
                    credits: { enabled: false },
                    navigation: {
                        buttonOptions: {
                            theme: { fill: null }
                        }
                    },
                    title: { text: maptitle },
                    xAxis: {
                        title: { text: 'Delta RA' },
                        allowDecimals: false,
                        min: 0,
                        max: cubeside - 1,
                        minorGridLineWidth: 0
                    },
                    yAxis: {
                        title: { text: 'Delta DEC' },
                        allowDecimals: false,
                        min: 0,
                        max: cubeside - 1,
                        endontick: false
                    },
                    colorAxis: { min: 0, max: 30,
                        // minColor: 'rgba(255,255,255,0)',
                        minColor: '#00BFFF',
                        maxColor: '#000080',
                        reversed: false,
                        labels: { align: 'right' }
                    },
                    legend: { align: 'right',
                        layout: 'vertical',
                        margin: 0,
                        verticalAlign: 'bottom',
                        y: -53,
                        symbolHeight: 380,
                        title: { text: 'Flux' }
                    },
                    tooltip: {
                        formatter: function formatter() {
                            return '<br>(' + this.point.x + ', ' + this.point.y + '): <b>' + this.point.value + '</b> Halphas <br>';
                        }
                    },
                    plotOptions: {
                        series: {
                            events: {
                                click: function click(event) {
                                    _this.getSpaxelFromMap(event);
                                }
                            }
                        }
                    },
                    series: [{
                        type: "heatmap",
                        name: "Halphas",
                        borderWidth: 0,
                        data: myjson,
                        dataLabels: { enabled: false }
                    }]
                });
                chart = $container.highcharts();
                if (chartWidth) {
                    if (chartWidth > 400) {
                        var chartWidth = 400;
                        var chartHeight = 400;
                    } else {
                        var chartHeight = chartWidth;
                    }
                    chart.setSize(chartWidth, chartHeight);
                }
                /*$('<button>+</button>').insertBefore($container).click(function () {
                    //chartWidth *= 1.1;
                    chartWidth = chartHeight;
                    chart.setSize(chartWidth, chartHeight);
                });*/
            });
        }
    }, {
        key: 'getSpaxelFromMap',
        value: function getSpaxelFromMap(event) {
            var keys = ['plateifu', 'x', 'y'];
            var form = m.utils.buildForm(keys, this.plateifu, event.point.x, event.point.y);
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
    }, {
        key: 'getSpaxel',


        // Retrieves a new Spaxel from the server based on a given mouse position
        value: function getSpaxel(event) {
            var map = event.map;
            var mousecoords = event.coordinate;
            var keys = ['plateifu', 'image', 'imwidth', 'imheight', 'mousecoords'];
            var form = m.utils.buildForm(keys, this.plateifu, this.image, this.olmap.imwidth, this.olmap.imheight, mousecoords);
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
    }, {
        key: 'toggleInteract',


        // Toggle the interactive OpenLayers map and Dygraph spectra
        value: function toggleInteract(image, map, spaxel, maptitle, spectitle, chartWidth) {
            if (this.togglediv.hasClass('active')) {
                // Turning Off
                this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
                this.togglediv.button('reset');
                this.dynamicdiv.hide();
                this.staticdiv.show();
            } else {
                // Turning On
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
                    console.log('image empty');
                    this.initOpenLayers(image);
                }
                // load the map if div is empty
                if (mapempty) {
                    this.initHeatmap(map, spaxel, maptitle, spectitle, 'mapdiv', chartWidth);
                    this.initHeatmap(map, spaxel, maptitle, spectitle, 'mapdiv2', chartWidth);
                    this.initHeatmap(map, spaxel, maptitle, spectitle, 'mapdiv3', chartWidth);
                }
            }
        }
    }, {
        key: 'initFlagPopovers',


        //  Initialize the Quality and Target Popovers
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
    }]);

    return Galaxy;
}();
