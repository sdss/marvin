/*
* @Author: Brian Cherinka
* @Date:   2016-08-30 11:28:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-08-30 13:59:35
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
            // this.colorNoData(Highcharts);
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

        // Get a slice of the mapdata array

    }, {
        key: 'getRange',
        value: function getRange(axis) {
            var axis = axis === undefined ? 0 : axis;
            var range = this.data.map(function (elt) {
                return elt[axis];
            });
            return range;
        }

        // return the min and max of a range

    }, {
        key: 'getMinMax',
        value: function getMinMax(range) {
            var range = range === undefined ? this.getRange() : range;
            var min = Math.min.apply(null, range);
            var max = Math.max.apply(null, range);
            return [min, max];
        }
    }, {
        key: 'interpColors',
        value: function interpColors(range) {
            var spread = range[1] - range[0];
            var midpt = (range[0] + range[1]) / 2;
            var f = chroma.scale();
            /*var data = this.data.map(function(el) {return ((el[2] - midpt) / spread) + 0.5});
            //{return ((element[2] - midpt) / spread) + 0.5});
            console.log('this.data', this.data);
            console.log('data', data);
            var colors = data.map(f);
            colors = colors.map(function(x) {return chroma(x).hex()});
            console.log('colors', colors);
            */

            var assignColor = function assignColor(el) {
                var normedVal = (el[2] - midpt) / spread + 0.5;
                var colorRGB = f(normedVal);
                var colorHex = chroma(colorRGB).hex();
                var spaxel = [{
                    x: el[0],
                    y: el[1],
                    value: el[2],
                    color: colorHex
                }];
                return spaxel;
            };
            var data = new Array();
            for (var ii = 0; ii < this.data.length; ii++) {
                data.push(assignColor(this.data[ii]));
            }
            return data;
        }
    }, {
        key: 'addNull',
        value: function addNull(x) {
            var x_nulls = x.map(function (el) {
                if (el[2] < 0.01) {
                    el[2] = null;
                };
                return el;
            });
            return x_nulls;
        }
    }, {
        key: 'colorNoData',
        value: function colorNoData(H) {
            H.wrap(H.ColorAxis.prototype, 'toColor', function (proceed, value, point) {
                if (value < 5) return '#FF00FF'; // My color
                else return proceed.apply(this, Array.prototype.slice.call(arguments, 1)); // Normal coloring
            });
        }

        // initialize the heat map

    }, {
        key: 'initMap',
        value: function initMap() {
            // set the galaxy class self to a variable
            var _galthis = this.galthis;

            // get the ranges
            var range = this.getRange(0);
            var zscale = this.getRange(2);

            // get the min and max of the ranges
            var xmin, xmax, zmin, zmax;

            var _getMinMax = this.getMinMax(range);

            var _getMinMax2 = _slicedToArray(_getMinMax, 2);

            xmin = _getMinMax2[0];
            xmax = _getMinMax2[1];

            // set the colors of each spaxel
            // var spaxColors = this.interpColors([zmin, zmax]);
            //console.log('spaxColors', spaxColors);
            //console.log('BEFORE this.data', this.data.length, this.data[0].length, this.data[0]);
            //var data = this.interpColors([zmin, zmax]);
            //console.log('AFTER data', data.length, data[0].length, data[0][0]);

            var _getMinMax3 = this.getMinMax(zscale);

            var _getMinMax4 = _slicedToArray(_getMinMax3, 2);

            zmin = _getMinMax4[0];
            zmax = _getMinMax4[1];
            var data = this.addNull(this.data);
            console.log('AFTER data', data);

            this.mapdiv.highcharts({
                chart: {
                    type: 'heatmap',
                    marginTop: 40,
                    marginBottom: 80,
                    plotBorderWidth: 1
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
                    min: xmin,
                    max: xmax,
                    tickInterval: 1
                },
                yAxis: {
                    title: { text: 'Spaxel Y' },
                    min: xmin,
                    max: xmax,
                    tickInterval: 1,
                    endontick: false
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
                        nullColor: '#FF00FF'
                    }
                },
                // colors: spaxColors,
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
