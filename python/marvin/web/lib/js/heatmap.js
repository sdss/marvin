/*
* @Author: Brian Cherinka
* @Date:   2016-08-30 11:28:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-09-23 01:13:51
*/

'use strict';

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var HeatMap = function () {

    // Constructor
    function HeatMap(mapdiv, data, title, galthis) {
        _classCallCheck(this, HeatMap);

        if (data === undefined) {
            console.error('Must specify input map data to initialize a HeatMap!');
        } else if (mapdiv === undefined) {
            console.error('Must specify an input mapdiv to initialize a HeatMap');
        } else {
            this.mapdiv = mapdiv; // div element for map
            this.data = data; // map data
            this.title = title; // map title
            this.galthis = galthis; //the self of the Galaxy class
            this.parseTitle();
            this.initMap();
            this.setColorNoData(this, Highcharts);
        }
    }

    _createClass(HeatMap, [{
        key: 'print',


        // test print
        value: function print() {
            console.log('We are now printing heatmap for ', this.title);
        }
    }, {
        key: 'parseTitle',


        // Parse the heatmap title into category, parameter, channel
        // e.g. 7443-1901: emline_gflux_ha-6564
        value: function parseTitle() {
            var _title$split = this.title.split(':');

            var _title$split2 = _slicedToArray(_title$split, 2);

            var plateifu = _title$split2[0];
            var newtitle = _title$split2[1];

            var _newtitle$split = newtitle.split('_');

            var _newtitle$split2 = _slicedToArray(_newtitle$split, 3);

            this.category = _newtitle$split2[0];
            this.parameter = _newtitle$split2[1];
            this.channel = _newtitle$split2[2];
        }

        // Get range of x (or y) data and z (DAP property) data

    }, {
        key: 'getRange',
        value: function getRange() {
            var xylength = this.data['values'].length;
            var xyrange = Array.apply(null, { length: xylength }).map(Number.call, Number);
            var zrange = [].concat.apply([], this.data['values']);
            return [xyrange, zrange];
        }

        // return the min and max of a range

    }, {
        key: 'getMinMax',
        value: function getMinMax(range) {
            // var range = (range === undefined) ? this.getRange() : range;
            var min = Math.min.apply(null, range);
            var max = Math.max.apply(null, range);
            return [min, max];
        }
    }, {
        key: 'setNull',
        value: function setNull(x) {
            var values = x.values;
            var ivar = x.ivar;
            var mask = x.mask;

            var xyz = Array();
            console.log('ivar', ivar);

            for (var ii = 0; ii < values.length; ii++) {
                for (var jj = 0; jj < values.length; jj++) {
                    var val = values[ii][jj];

                    var noValue = mask[ii][jj] && Math.pow(2, 0);
                    var badValue = mask[ii][jj] && Math.pow(2, 5);
                    var mathError = mask[ii][jj] && Math.pow(2, 6);
                    var badFit = mask[ii][jj] && Math.pow(2, 7);
                    var doNotUse = mask[ii][jj] && Math.pow(2, 30);
                    var noData = noValue || badValue || mathError || badFit || doNotUse;

                    var signalToNoise = val * Math.sqrt(ivar[ii][jj]);
                    var signalToNoiseThreshold = 1.;

                    if (noData) {
                        val = 'no-data';
                    } else if (signalToNoise < signalToNoiseThreshold) {
                        val = null;
                    };
                    xyz.push([ii, jj, val]);
                };
            };
            return xyz;
        }
    }, {
        key: 'setColorNoData',
        value: function setColorNoData(_this, H) {
            H.wrap(H.ColorAxis.prototype, 'toColor', function (proceed, value, point) {
                if (value === 'no-data') {
                    return 'rgba(0,0,0,0)'; // '#A8A8A8';
                } else return proceed.apply(this, Array.prototype.slice.call(arguments, 1));
            });
        }

        // initialize the heat map

    }, {
        key: 'initMap',
        value: function initMap() {
            // set the galaxy class self to a variable
            var _galthis = this.galthis;

            // get the ranges
            //var range  = this.getXRange();
            var xyrange, zrange;

            // get the min and max of the ranges
            var _getRange = this.getRange();

            var _getRange2 = _slicedToArray(_getRange, 2);

            xyrange = _getRange2[0];
            zrange = _getRange2[1];
            var xymin, xymax, zmin, zmax;

            var _getMinMax = this.getMinMax(xyrange);

            var _getMinMax2 = _slicedToArray(_getMinMax, 2);

            xymin = _getMinMax2[0];
            xymax = _getMinMax2[1];

            var _getMinMax3 = this.getMinMax(zrange);

            var _getMinMax4 = _slicedToArray(_getMinMax3, 2);

            zmin = _getMinMax4[0];
            zmax = _getMinMax4[1];


            var data = this.setNull(this.data);

            this.mapdiv.highcharts({
                chart: {
                    type: 'heatmap',
                    marginTop: 40,
                    marginBottom: 80,
                    plotBorderWidth: 1,
                    backgroundColor: null,
                    plotBackgroundColor: '#A8A8A8'
                },
                credits: { enabled: false },
                title: { text: this.title },
                navigation: {
                    buttonOptions: {
                        theme: { fill: null }
                    }
                },
                xAxis: {
                    title: { text: 'Spaxel X' },
                    minorGridLineWidth: 0,
                    min: xymin,
                    max: xymax,
                    tickInterval: 1,
                    tickLength: 0
                },
                yAxis: {
                    title: { text: 'Spaxel Y' },
                    min: xymin,
                    max: xymax,
                    tickInterval: 1,
                    endOnTick: false,
                    gridLineWidth: 0
                },
                colorAxis: {
                    min: Math.floor(zmin),
                    max: Math.ceil(zmax),
                    minColor: '#00BFFF',
                    maxColor: '#000080',
                    labels: { align: 'right' },
                    reversed: false
                },
                plotOptions: {
                    heatmap: {
                        nullColor: 'url(#custom-pattern)' //'#A8A8A8'
                    }
                },
                defs: {
                    patterns: [{
                        width: 3,
                        height: 3,
                        'id': 'custom-pattern',
                        'path': {
                            // I *think* M and L define the start and end points of line segments of the
                            // pattern in units of the width and height, which both default to 10. To
                            // change the density of the pattern hatching, decrease the width and height
                            // and then scale down the "d" values accorindingly.
                            // The second and third set of M and L coordinates color in the upper right
                            // and lower left corners of the box to make the line segments of the
                            // adjacent boxes look continuous. This isn't needed for the vertical or
                            // horizontal hatching.
                            // d: 'M 0 0 L 10 10 M 9 -1 L 11 1 M -1 9 L 1 11',
                            // d: 'M 0 0 L 7 7 M 6 -1 L 8 1 M -1 6 L 1 8',
                            d: 'M 0 0 L 3 3 M 0 3 L 3 0',
                            stroke: 'white', // '#A8A8A8',
                            strokeWidth: 0.3
                        }
                    }]
                },
                legend: {
                    align: 'right',
                    layout: 'vertical',
                    verticalAlign: 'middle',
                    title: { text: this.parameter }
                },
                tooltip: {
                    formatter: function formatter() {
                        return '<br>(' + this.point.x + ', ' + this.point.y + '): <b>' + this.point.value + '</b><br>';
                    }
                },
                series: [{
                    type: "heatmap",
                    name: "halphas",
                    data: data,
                    dataLabels: { enabled: false },
                    events: {
                        click: function click(event) {
                            _galthis.getSpaxel(event);
                        }
                    }
                }]
            });
        }
    }]);

    return HeatMap;
}();
