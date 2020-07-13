/*
* @Author: Brian Cherinka
* @Date:   2016-12-09 01:38:32
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 01:03:57
*/

//jshint esversion: 6
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
            this.createTitleOverlays();
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
                },
                xrev: false,
                yrev: false
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
            this.plotdiv.empty();
            this.chart = Highcharts.chart(this.plotdiv.attr('id'), {
                chart: {
                    type: 'scatter',
                    zoomType: 'xy',
                    backgroundColor: '#F5F5F5',
                    plotBackgroundColor: '#F5F5F5'
                },
                title: {
                    text: null //this.cfg.title
                },
                xAxis: {
                    title: {
                        enabled: true,
                        text: this.cfg.xtitle
                    },
                    startOnTick: true,
                    endOnTick: true,
                    showLastLabel: true,
                    reversed: this.cfg.xrev,
                    id: this.cfg.xtitle.replace(/\s/g, '').toLowerCase() + '-axis'
                },
                yAxis: {
                    title: {
                        text: this.cfg.ytitle
                    },
                    gridLineWidth: 0,
                    reversed: this.cfg.yrev,
                    id: this.cfg.ytitle.replace(/\s/g, '').toLowerCase() + '-axis'
                },
                legend: {
                    layout: 'vertical',
                    align: 'left',
                    verticalAlign: 'top',
                    x: 75,
                    y: 20,
                    title: {
                        text: 'Drag Me'
                    },
                    floating: true,
                    draggable: true,
                    backgroundColor: Highcharts.theme && Highcharts.theme.legendBackgroundColor || '#FFFFFF',
                    borderWidth: 1
                },
                boost: {
                    useGPUTranslations: true,
                    usePreAllocated: true
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

        // Create Axis Title Overlays for Drag and Drop highlighting

    }, {
        key: 'createTitleOverlays',
        value: function createTitleOverlays() {
            this.overgroup = this.chart.renderer.g().add();
            this.overheight = 20;
            this.overwidth = 100;
            this.overedge = 5;

            // styling
            this.overcolor = 'rgba(255,0,0,0.5)';
            this.overborder = 'black';
            this.overbwidth = 2;
            this.overzindex = 3;

            var xtextsvg = this.chart.xAxis[0].axisTitle.element;
            var xtextsvg_x = xtextsvg.getAttribute('x');
            var xtextsvg_y = xtextsvg.getAttribute('y');

            var ytextsvg = this.chart.yAxis[0].axisTitle.element;
            var ytextsvg_x = ytextsvg.getAttribute('x');
            var ytextsvg_y = ytextsvg.getAttribute('y');

            this.yover = this.chart.renderer.rect(ytextsvg_x - (this.overheight / 2. + 3), ytextsvg_y - this.overwidth / 2., this.overheight, this.overwidth, this.overedge).attr({
                'stroke-width': this.overbwidth,
                stroke: this.overborder,
                fill: this.overcolor,
                zIndex: this.overzindex
            }).add(this.overgroup);

            this.xover = this.chart.renderer.rect(xtextsvg_x - this.overwidth / 2., xtextsvg_y - (this.overheight / 2 + 3), this.overwidth, this.overheight, this.overedge).attr({
                'stroke-width': this.overbwidth,
                stroke: this.overborder,
                fill: this.overcolor,
                zIndex: this.overzindex
            }).add(this.overgroup);
            this.overgroup.hide();
        }
    }]);

    return Scatter;
}();
