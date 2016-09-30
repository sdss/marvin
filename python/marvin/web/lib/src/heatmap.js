/*
* @Author: Brian Cherinka
* @Date:   2016-08-30 11:28:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-09-30 10:56:42
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

                var noValue = (mask[ii][jj] && Math.pow(2, 0));
                var badValue = (mask[ii][jj] && Math.pow(2, 5));
                var mathError = (mask[ii][jj] && Math.pow(2, 6));
                var badFit = (mask[ii][jj] && Math.pow(2, 7));
                var doNotUse = (mask[ii][jj] && Math.pow(2, 30));
                var noData = (noValue || badValue || mathError || badFit || doNotUse);

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

    setColorNoData(_this, H) {
        H.wrap(H.ColorAxis.prototype, 'toColor', function (proceed, value, point) {
            if(value === 'no-data') {
                return 'rgba(0,0,0,0)';  // '#A8A8A8';
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
                min: Math.floor(zmin),
                max: Math.ceil(zmax),
                minColor: '#00BFFF',
                maxColor: '#000080',
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

