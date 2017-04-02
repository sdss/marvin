/*
* @Author: Brian Cherinka
* @Date:   2016-12-13 09:49:30
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 01:18:52
*/

//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var BoxWhisker = function () {

    // Constructor
    function BoxWhisker(id, data, options) {
        _classCallCheck(this, BoxWhisker);

        if (data === undefined) {
            console.error('Must specify input plot data to initialize a BoxWhisker!');
        } else if (id === undefined) {
            console.error('Must specify an input plotdiv to initialize a BoxWhisker');
        } else {
            this.plotdiv = id; // div element for map
            this.plotid = '#' + this.plotdiv.attr('id');
            this.tooltip = '#d3tooltip';
            this.data = data; // map data
            //this.title = title; // map title
            //this.origthis = galthis; //the self of the Galaxy class
            //this.parseTitle();
            this.setOptions(options);
            this.initBoxplot();
        }
    }

    // test print


    _createClass(BoxWhisker, [{
        key: 'print',
        value: function print() {
            console.log('We are now printing boxwhisker for ', this.cfg.title);
        }

        // sets the options

    }, {
        key: 'setOptions',
        value: function setOptions(options) {
            this.margin = { top: 10, right: 50, bottom: 40, left: 50 };
            // create the default options
            this.cfg = {
                title: 'BoxWhisker Title',
                origthis: null,
                width: 120 - this.margin.left - this.margin.right,
                height: 500 - this.margin.top - this.margin.bottom
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

        // Compute the IQR

    }, {
        key: 'iqr',
        value: function iqr(k) {
            return function (d, index) {
                var q1 = d.quartiles[0],
                    q3 = d.quartiles[2],
                    iqr = (q3 - q1) * k,
                    i = -1,
                    j = d.length;
                while (d[++i] < q1 - iqr) {}
                while (d[--j] > q3 + iqr) {}
                return [i, j];
            };
        }

        // initialize the D3 box and whisker plot

    }, {
        key: 'initBoxplot',
        value: function initBoxplot() {

            // // Define the div for the tooltip
            // let tooltip = d3.select(this.tooltip).append("div")
            //     .attr("class", "tooltip")
            //     .style("opacity", 0);

            // Make the chart
            var chart = d3.box().whiskers(this.iqr(1.5)).width(this.cfg.width).height(this.cfg.height);

            // load in the data and create the box plot
            var svg = d3.select(this.plotid).selectAll("svg").data(this.data).enter().append("svg").attr("class", "box").attr("width", this.cfg.width + this.margin.left + this.margin.right).attr("height", this.cfg.height + this.margin.bottom + this.margin.top).append("g").attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")").call(chart);
        }
    }]);

    return BoxWhisker;
}();
