/*
* @Author: Brian Cherinka
* @Date:   2016-08-30 11:28:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-08-30 13:59:35
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

    // Get a slice of the mapdata array
    getRange(axis) {
        var axis = (axis === undefined ) ? 0 : axis;
        var range  = this.data.map(function(elt) { return elt[axis]; });
        return range;
    }

    // return the min and max of a range
    getMinMax(range) {
        var range = (range === undefined) ? this.getRange() : range;
        var min = Math.min.apply(null, range);
        var max = Math.max.apply(null, range);
        return [min, max];
    }

    // initialize the heat map
    initMap() {
        // set the galaxy class self to a variable
        var _galthis = this.galthis;

        // get the ranges
        var range  = this.getRange(0);
        var zscale  = this.getRange(2);

        // get the min and max of the ranges
        var xmin, xmax, zmin, zmax;
        [xmin, xmax] = this.getMinMax(range);
        [zmin, zmax] = this.getMinMax(zscale);

        this.mapdiv.highcharts({
            chart: {
                type: 'heatmap',
                marginTop: 40,
                marginBottom: 80,
                plotBorderWidth: 1
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
                min: xmin,
                max: xmax,
                tickInterval: 1
            },
            yAxis:{
                title: {text: 'Spaxel Y'},
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
                labels: {align: 'right'}
            },
            legend: {align: 'right',
                layout: 'vertical',
                verticalAlign: 'middle',
                title: {text: this.parameter},
                reversed: true
            },
            tooltip: {
                formatter: function () {
                    return '<br>('+this.point.x+', '+this.point.y+'): <b>'+this.point.value+'</b><br>';
                }
            },
            series:[{
                type: "heatmap",
                name: "halphas",
                data: this.data,
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

