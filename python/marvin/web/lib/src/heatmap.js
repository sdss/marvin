/*
* @Author: Brian Cherinka
* @Date:   2016-08-30 11:28:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-10-11 13:13:52
*/

'use strict';

class HeatMap {

    // Constructor
    constructor(mapdiv, data, title, galthis) {
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

    };

    // test print
    print() {
        console.log('We are now printing heatmap for ', this.title);
    };

    // Parse the heatmap title into category, parameter, channel
    // e.g. 7443-1901: emline_gflux_ha-6564
    parseTitle() {
        var [plateifu, newtitle] = this.title.split(':');
        [this.category, this.parameter, this.channel] = newtitle.split('_');
    }

    // Get range of x (or y) data and z (DAP property) data
    getRange(){
        var xylength  = this.data['values'].length;
        var xyrange = Array.apply(null, {length: xylength}).map(Number.call, Number);
        var zrange = [].concat.apply([], this.data['values']);
        return [xyrange, zrange];
    }

    // Filter out null and no-data from z (DAP prop) data
    filterRange(z) {
        if (z !== undefined && typeof(z) === 'number' && !isNaN(z)) {
            return true;
        } else {
            return false;
        }
    }

    // return the min and max of a range
    getMinMax(range) {
        // var range = (range === undefined) ? this.getRange() : range;
        var min = Math.min.apply(null, range);
        var max = Math.max.apply(null, range);
        return [min, max];
    }

    setNull(x) {
        var values = x.values;
        var ivar = x.ivar;
        var mask = x.mask;

        var xyz = Array();

        for (var ii=0; ii < values.length; ii++) {
            for (var jj=0; jj < values.length; jj++){
                var val = values[ii][jj];

                if (mask !== null) {
                    var noValue = (mask[ii][jj] && Math.pow(2, 0));
                    var badValue = (mask[ii][jj] && Math.pow(2, 5));
                    var mathError = (mask[ii][jj] && Math.pow(2, 6));
                    var badFit = (mask[ii][jj] && Math.pow(2, 7));
                    var doNotUse = (mask[ii][jj] && Math.pow(2, 30));
                    //var noData = (noValue || badValue || mathError || badFit || doNotUse);
                    var noData = noValue;
                    var badData = (badValue || mathError || badFit || doNotUse);
                } else {
                    noData == null;
                    badData == null;
                }

                if (ivar !== null) {
                    var signalToNoise = val * Math.sqrt(ivar[ii][jj]);
                    var signalToNoiseThreshold = 1.;
                }

                // value types
                // val=no-data => gray color
                // val=null => hatch area
                // val=low-sn => nothing at the moment

                if (noData) {
                    // for data that is outside the range "nocov" mask
                    val = 'no-data';
                } else if (badData) {
                    // for data that is bad - masked in some way
                    val = null;
                } else if (ivar !== null && (signalToNoise < signalToNoiseThreshold)) {
                    // for data that is low S/N
                   var g = null ; //val = 'low-sn';
                } else if (ivar === null) {
                    // for data with no mask or no inverse variance extensions
                    if (this.title.search('binid') !== -1) {
                        // for binid extension only, set -1 values to no data
                        val = (val == -1 ) ? 'no-data' : val;
                    } else if (val === 0.0) {
                        // set zero values to no-data
                        val = 'no-data';
                    }
                };
                xyz.push([ii, jj, val]);
            };
        };
        return xyz;
    }

    setColorNoData(_this, H) {
        H.wrap(H.ColorAxis.prototype, 'toColor', function (proceed, value, point) {
            if (value === 'no-data') {
                // make gray color
                return 'rgba(0,0,0,0)';  // '#A8A8A8';
            }
            else if (value === 'low-sn') {
                // make light blue with half-opacity == muddy blue-gray
                return 'rgba(0,191,255,0.5)'; //'#7fffd4';
            }
            else
                return proceed.apply(this, Array.prototype.slice.call(arguments, 1));
        });
    }

    // initialize the heat map
    initMap() {
        // set the galaxy class self to a variable
        var _galthis = this.galthis;

        // get the ranges
        //var range  = this.getXRange();
        var xyrange, zrange;
        [xyrange, zrange]  = this.getRange();

        // get the min and max of the ranges
        var xymin, xymax, zmin, zmax;
        [xymin, xymax] = this.getMinMax(xyrange);
        [zmin, zmax] = this.getMinMax(zrange);

        // set null data and create new zrange, min, and max
        var data = this.setNull(this.data);
        zrange = data.map(function(o){return o[2];});
        zrange = zrange.filter(this.filterRange);
        [zmin, zmax] = this.getMinMax(zrange);
        console.log('new zrange', zrange);
        console.log('new zminmax', zmin, zmax);

        // make the highcharts
        this.mapdiv.highcharts({
            chart: {
                type: 'heatmap',
                marginTop: 40,
                marginBottom: 80,
                plotBorderWidth: 1,
                backgroundColor: null,
                plotBackgroundColor: '#A8A8A8'
            },
            credits: {enabled: false},
            title: {text: this.title},
            navigation: {
                buttonOptions: {
                    theme: {fill: null}
                }
            },
            xAxis: {
                title: {text: 'Spaxel X'},
                minorGridLineWidth: 0,
                min: xymin,
                max: xymax,
                tickInterval: 1,
                tickLength: 0
            },
            yAxis:{
                title: {text: 'Spaxel Y'},
                min: xymin,
                max: xymax,
                tickInterval: 1,
                endOnTick: false,
                gridLineWidth: 0
            },
            colorAxis: {
                min: zmin,
                max: zmax,
                minColor: (zmin >= 0.0) ? '#00BFFF' : '#ff3030',
                maxColor: '#000080',
                //stops: [[0, '#ff3030'], [0.5, '#f8f8ff'], [1, '#000080']],
                labels: {align: 'right'},
                reversed: false
            },
            plotOptions: {
                heatmap:{
                    nullColor: 'url(#custom-pattern)'  //'#A8A8A8'
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
                        strokeWidth: 0.3,
                        // fill: 'rgba(255, 255, 255, 1)'  // 'rgba(168, 168, 168, 0.3)'
                    }
                }]
            },
            legend: {
                align: 'right',
                layout: 'vertical',
                verticalAlign: 'middle',
                title: {text: this.parameter},
            },
            tooltip: {
                formatter: function () {
                    return '<br>('+this.point.x+', '+this.point.y+'): <b>'+this.point.value+'</b><br>';
                }
            },
            series:[{
                type: "heatmap",
                name: "halphas",
                data: data,
                dataLabels: {enabled: false},
                events: {
                    click: function (event) {
                        _galthis.getSpaxel(event);
                    }
                }
            }]
        });
    }

}

