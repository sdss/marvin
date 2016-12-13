/*
* @Author: Brian Cherinka
* @Date:   2016-12-09 01:38:32
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-12-13 09:51:10
*/

'use strict';

// Creates a Scatter Plot Highcharts Object

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Scatter = function () {

    // Constructor
    function Scatter(id, data, options) {
        _classCallCheck(this, Scatter);

        if (data === undefined) {
            console.error('Must specify input plot data to initialize a ScatterPlot!');
        } else if (id === undefined) {
            console.error('Must specify an input plotdiv to initialize a ScatterPlot');
        } else {
            this.plotdiv = id; // div element for map
            this.data = data; // map data
            //this.title = title; // map title
            //this.origthis = galthis; //the self of the Galaxy class
            //this.parseTitle();
            this.setOptions(options);
            this.initChart();
        }
    }

    // test print


    _createClass(Scatter, [{
        key: 'print',
        value: function print() {
            console.log('We are now printing scatter for ', this.cfg.title);
        }

        // sets the options

    }, {
        key: 'setOptions',
        value: function setOptions(options) {
            // create the default options
            this.cfg = {
                title: 'Scatter Title',
                origthis: null,
                xtitle: 'X-Axis',
                ytitle: 'Y-Axis',
                galaxy: {
                    name: 'Galaxy'
                },
                altseries: {
                    name: null,
                    data: null
                }
            };

            //Put all of the options into a variable called cfg
            if ('undefined' !== typeof options) {
                for (var i in options) {
                    if ('undefined' !== typeof options[i]) {
                        this.cfg[i] = options[i];
                    }
                }
            }
        }

        // initialize the chart

    }, {
        key: 'initChart',
        value: function initChart() {
            console.log('init plotdiv', this.plotdiv.attr('id'));
            this.plotdiv.empty();
            this.chart = Highcharts.chart(this.plotdiv.attr('id'), {
                chart: {
                    type: 'scatter',
                    zoomType: 'xy',
                    backgroundColor: '#F5F5F5',
                    plotBackgroundColor: '#F5F5F5'
                },
                title: {
                    text: this.cfg.title
                },
                xAxis: {
                    title: {
                        enabled: true,
                        text: this.cfg.xtitle
                    },
                    startOnTick: true,
                    endOnTick: true,
                    showLastLabel: true,
                    id: this.cfg.xtitle.replace(/\s/g, '').toLowerCase() + '-axis'
                },
                yAxis: {
                    title: {
                        text: this.cfg.ytitle
                    },
                    gridLineWidth: 0,
                    id: this.cfg.ytitle.replace(/\s/g, '').toLowerCase() + '-axis'
                },
                legend: {
                    layout: 'vertical',
                    align: 'left',
                    verticalAlign: 'top',
                    x: 100,
                    y: 70,
                    floating: true,
                    backgroundColor: Highcharts.theme && Highcharts.theme.legendBackgroundColor || '#FFFFFF',
                    borderWidth: 1
                },
                plotOptions: {
                    scatter: {
                        marker: {
                            radius: 5,
                            states: {
                                hover: {
                                    enabled: true,
                                    lineColor: 'rgb(100,100,100)'
                                }
                            }
                        },
                        states: {
                            hover: {
                                marker: {
                                    enabled: false
                                }
                            }
                        },
                        tooltip: {
                            headerFormat: '<b>{series.name}</b><br>',
                            pointFormat: '({point.x}, {point.y})'
                        }
                    }
                },
                series: [{
                    name: this.cfg.altseries.name,
                    color: 'rgba(70,130,180,0.4)',
                    data: this.cfg.altseries.data,
                    turboThreshold: 0,
                    marker: {
                        radius: 2,
                        symbol: 'circle'
                    },
                    tooltip: {
                        headerFormat: '<b>{series.name}: {point.key}</b><br>' }

                }, {
                    name: this.cfg.galaxy.name,
                    color: 'rgba(255, 0, 0, 1)',
                    data: this.data,
                    marker: { symbol: 'circle', radius: 5 }
                }]
            });
        }
    }]);

    return Scatter;
}();
