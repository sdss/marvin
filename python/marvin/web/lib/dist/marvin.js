/*
* @Author: Brian Cherinka
* @Date:   2016-12-13 09:41:40
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 01:20:05
*/

// Using Mike Bostocks box.js code
// https://bl.ocks.org/mbostock/4061502

// This has been modified by me to accept data as a
// a list of objects in the format of
// data = [ {'value': number, 'title': string_name, 'sample': array of points}, ..]
// This allows to display box and whisker plots of an array of data
// and overplot a single value within this space

// Dec-13-2016 - converted to D3 v4

//jshint esversion: 6
'use strict';

function iqr(k) {
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

function boxWhiskers(d) {
    return [0, d.length - 1];
}

function boxQuartiles(d) {
    return [d3.quantile(d, .25), d3.quantile(d, .5), d3.quantile(d, .75)];
}

function getTooltip() {
    var tooltip = d3.select('body').append("div").attr("class", "tooltip").style("opacity", 0);
    return tooltip;
}

// Inspired by http://informationandvisualization.de/blog/box-plot
d3.box = function () {
    var width = 1,
        height = 1,
        duration = 0,
        domain = null,
        value = Number,
        whiskers = boxWhiskers,
        quartiles = boxQuartiles,
        showLabels = true,
        x1 = null,
        // the x1 variable here represents the y-axis
    x0 = null,
        // the old y-axis
    tickFormat = null;

    var tooltip = getTooltip();

    // For each small multiple…
    function box(g) {
        g.each(function (d, i) {
            var origd = d;
            d = d.sample.map(value).sort(d3.ascending);
            var g = d3.select(this),
                n = d.length,
                min = d[0],
                max = d[n - 1];

            // Compute quartiles. Must return exactly 3 elements.
            var quartileData = d.quartiles = quartiles(d);
            var q10 = d3.quantile(d, .10);
            var q90 = d3.quantile(d, .90);
            var myiqr = quartileData[2] - quartileData[0];

            // compute the 2.5*iqr indices and values
            var iqr25inds = iqr(2.5).call(this, d, i);
            var iqr25data = iqr25inds.map(function (i) {
                return d[i];
            });

            // Compute whiskers. Must return exactly 2 elements, or null.
            var whiskerIndices = whiskers && whiskers.call(this, d, i),
                whiskerData = whiskerIndices && whiskerIndices.map(function (i) {
                return d[i];
            });

            // Compute outliers. If no whiskers are specified, all data are "outliers".
            // We compute the outliers as indices, so that we can join across transitions!
            var outlierIndices = whiskerIndices ? d3.range(0, whiskerIndices[0]).concat(d3.range(whiskerIndices[1] + 1, n)) : d3.range(n);

            // Compute the new x-scale.
            var q50 = quartileData[1];
            var zero = Math.max(iqr25data[1] - q50, q50 - iqr25data[0]); //rescales the axis to center each plot on the median
            var diff = Math.min(max - whiskerData[1], min - whiskerData[0]);

            x1 = d3.scaleLinear().domain([q50 - zero, q50, q50 + zero]).range([height, height / 2, 0]);
            //.domain([min,max])
            //.range([height,0]);

            // Retrieve the old x-scale, if this is an update.
            x0 = this.__chart__ || d3.scaleLinear().domain([0, Infinity]).range(x1.range());

            // Stash the new scale.
            this.__chart__ = x1;

            // Note: the box, median, and box tick elements are fixed in number,
            // so we only have to handle enter and update. In contrast, the outliers
            // and other elements are variable, so we need to exit them! Variable
            // elements also fade in and out.

            // Update center line: the vertical line spanning the whiskers.
            var center = g.selectAll("line.center").data(whiskerData ? [whiskerData] : []);

            center.enter().insert("line", "rect").attr("class", "center").attr("x1", width / 2).attr("y1", function (d) {
                return x0(d[0]);
            }).attr("x2", width / 2).attr("y2", function (d) {
                return x0(d[1]);
            }).style("opacity", 1e-6).transition().duration(duration).style("opacity", 1).attr("y1", function (d) {
                return x1(d[0]);
            }).attr("y2", function (d) {
                return x1(d[1]);
            });

            center.transition().duration(duration).style("opacity", 1).attr("y1", function (d) {
                return x1(d[0]);
            }).attr("y2", function (d) {
                return x1(d[1]);
            });

            center.exit().transition().duration(duration).style("opacity", 1e-6).attr("y1", function (d) {
                return x1(d[0]);
            }).attr("y2", function (d) {
                return x1(d[1]);
            }).remove();

            // Update innerquartile box.
            var box = g.selectAll("rect.box").data([quartileData]);

            box.enter().append("rect").attr("class", "box").attr("x", 0).attr("y", function (d) {
                return x0(d[2]);
            }).attr("width", width).attr("height", function (d) {
                return x0(d[0]) - x0(d[2]);
            }).transition().duration(duration).attr("y", function (d) {
                return x1(d[2]);
            }).attr("height", function (d) {
                return x1(d[0]) - x1(d[2]);
            });

            box.transition().duration(duration).attr("y", function (d) {
                return x1(d[2]);
            }).attr("height", function (d) {
                return x1(d[0]) - x1(d[2]);
            });

            // Update median line.
            var medianLine = g.selectAll("line.median").data([quartileData[1]]);

            medianLine.enter().append("line").attr("class", "median").attr("x1", 0).attr("y1", x0).attr("x2", width).attr("y2", x0).transition().duration(duration).attr("y1", x1).attr("y2", x1);

            medianLine.transition().duration(duration).attr("y1", x1).attr("y2", x1);

            // Update whiskers.
            var whisker = g.selectAll("line.whisker").data(whiskerData || []);

            whisker.enter().insert("line", "circle, text").attr("class", "whisker").attr("x1", 0).attr("y1", x0).attr("x2", width).attr("y2", x0).style("opacity", 1e-6).transition().duration(duration).attr("y1", x1).attr("y2", x1).style("opacity", 1);

            whisker.transition().duration(duration).attr("y1", x1).attr("y2", x1).style("opacity", 1);

            whisker.exit().transition().duration(duration).attr("y1", x1).attr("y2", x1).style("opacity", 1e-6).remove();

            // update datapoint circle
            if (origd.value) {

                var datapoint = g.selectAll('circle.datapoint').data([origd.value]);

                datapoint.enter().append('circle').attr('class', 'datapoint').attr('cx', width / 2).attr('cy', x1(origd.value)).attr("r", 5).style('fill', 'red').on("mouseover", function (d) {
                    tooltip.transition().duration(200).style("opacity", .9);
                    tooltip.html('<b> Value: ' + origd.value + '</b>').style("left", d3.event.pageX + "px").style("top", d3.event.pageY - 28 + "px");
                }).on("mouseout", function (d) {
                    tooltip.transition().duration(500).style("opacity", 0);
                });
            }

            // update title
            if (origd.title) {
                var title = g.selectAll('text.title').data([origd.title]);
                title.enter().append('text').text(origd.title).attr('x', width / 2).attr('y', height).attr('dy', 20).attr('text-anchor', 'middle').style('font-weight', 'bold').style('font-size', 15);
            }

            // Update outliers.
            var outlier = g.selectAll("circle.outlier").data(outlierIndices, Number);

            outlier.enter().insert("circle", "text").attr("class", "outlier").attr("r", 5).attr("cx", width / 2).attr("cy", function (i) {
                return x0(d[i]);
            }).style("opacity", 1e-6).on("mouseover", function (i) {
                tooltip.transition().duration(200).style("opacity", .9);
                tooltip.html('<b> Value: ' + d[i] + '</b>').style("left", d3.event.pageX + "px").style("top", d3.event.pageY - 28 + "px");
            }).on("mouseout", function (d) {
                tooltip.transition().duration(500).style("opacity", 0);
            }).transition().duration(duration).attr("cy", function (i) {
                return x1(d[i]);
            }).style("opacity", 1);

            // outlier.transition()
            //     .duration(duration)
            //     .attr("cy", function(i) { return x1(d[i]); })
            //     .style("opacity", 1);

            // outlier.exit().transition()
            //     .duration(duration)
            //     .attr("cy", function(i) { return x1(d[i]); })
            //     .style("opacity", 1e-6)
            //     .remove();

            // Compute the tick format.
            var format = tickFormat || x1.tickFormat(8);

            // Update box ticks.
            var boxTick = g.selectAll("text.box").data(quartileData);

            if (showLabels === true) {
                boxTick.enter().append("text").attr("class", "box").attr("dy", ".3em").attr("dx", function (d, i) {
                    return i & 1 ? 6 : -6;
                }).attr("x", function (d, i) {
                    return i & 1 ? width : 0;
                }).attr("y", x0).attr("text-anchor", function (d, i) {
                    return i & 1 ? "start" : "end";
                }).text(format).transition().duration(duration).attr("y", x1);
            }

            boxTick.transition().duration(duration).text(format).attr("y", x1);

            // Update whisker ticks. These are handled separately from the box
            // ticks because they may or may not exist, and we want don't want
            // to join box ticks pre-transition with whisker ticks post-.
            var whiskerTick = g.selectAll("text.whisker").data(whiskerData || []);

            whiskerTick.enter().append("text").attr("class", "whisker").attr("dy", ".3em").attr("dx", 6).attr("x", width).attr("y", x0).text(format).style("opacity", 1e-6).transition().duration(duration).attr("y", x1).style("opacity", 1);

            whiskerTick.transition().duration(duration).text(format).attr("y", x1).style("opacity", 1);

            whiskerTick.exit().transition().duration(duration).attr("y", x1).style("opacity", 1e-6).remove();
        });
        d3.timerFlush();
    }

    box.overlay = function (x) {
        if (!arguments.length) return overlay;
        overlay = x;
        return overlay;
    };

    box.x1 = function (x) {
        if (!arguments.length) return x1;
        return x1(x);
    };

    box.x0 = function (x) {
        if (!arguments.length) return x0;
        return x0(x);
    };

    box.width = function (x) {
        if (!arguments.length) return width;
        width = x;
        return box;
    };

    box.height = function (x) {
        if (!arguments.length) return height;
        height = x;
        return box;
    };

    box.tickFormat = function (x) {
        if (!arguments.length) return tickFormat;
        tickFormat = x;
        return box;
    };

    box.duration = function (x) {
        if (!arguments.length) return duration;
        duration = x;
        return box;
    };

    box.domain = function (x) {
        if (!arguments.length) return domain;
        domain = x === null ? x : d3.functor(x);
        return box;
    };

    box.value = function (x) {
        if (!arguments.length) return value;
        value = x;
        return box;
    };

    box.tooltip = function (x) {
        if (!arguments.length) return tooltip;
        tooltip = x;
        return tooltip;
    };

    box.whiskers = function (x) {
        if (!arguments.length) return whiskers;
        whiskers = x;
        return box;
    };

    box.showLabels = function (x) {
        if (!arguments.length) return showLabels;
        showLabels = x;
        return box;
    };

    box.quartiles = function (x) {
        if (!arguments.length) return quartiles;
        quartiles = x;
        return box;
    };

    return box;
};
;/*
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
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-29 09:29:24
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 01:11:43
*/

//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Carousel = function () {

    // Constructor
    function Carousel(cardiv, thumbs) {
        _classCallCheck(this, Carousel);

        this.carouseldiv = $(cardiv);
        this.thumbsdiv = thumbs !== undefined ? $(thumbs) : $('[id^=carousel-selector-]');

        // init the carousel
        this.carouseldiv.carousel({
            interval: 5000
        });

        // Event handlers
        this.thumbsdiv.on('click', this, this.handleThumbs);
        this.carouseldiv.on('slid.bs.carousel', this, this.updateText);
    }

    // Print


    _createClass(Carousel, [{
        key: 'print',
        value: function print() {
            console.log('I am Carousel!');
        }

        // Handle the carousel thumbnails

    }, {
        key: 'handleThumbs',
        value: function handleThumbs(event) {
            var _this = event.data;
            var id_selector = $(this).attr("id");
            try {
                var id = /-(\d+)$/.exec(id_selector)[1];
                //console.log(id_selector, id);
                _this.carouseldiv.carousel(parseInt(id));
            } catch (e) {
                console.log('MyCarousel: Regex failed!', e);
            }
        }

        // When carousel slides, auto update the text

    }, {
        key: 'updateText',
        value: function updateText(event) {
            var _this = event.data;
            var id = $('.item.active').data('slide-number');
            $('#carousel-text').html($('#slide-content-' + id).html());
        }
    }]);

    return Carousel;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 16:49:00
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2018-11-17 14:36:53
*/

//
// Javascript Galaxy object handling JS things for a single galaxy
//
//jshint esversion: 6
'use strict';

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var SpaxelError = function (_Error) {
    _inherits(SpaxelError, _Error);

    function SpaxelError(message) {
        _classCallCheck(this, SpaxelError);

        var _this2 = _possibleConstructorReturn(this, (SpaxelError.__proto__ || Object.getPrototypeOf(SpaxelError)).call(this, message));

        _this2.message = message;
        _this2.name = 'SpaxelError';
        _this2.status = -1;
        return _this2;
    }

    return SpaxelError;
}(Error);

var MapError = function (_Error2) {
    _inherits(MapError, _Error2);

    function MapError(message) {
        _classCallCheck(this, MapError);

        var _this3 = _possibleConstructorReturn(this, (MapError.__proto__ || Object.getPrototypeOf(MapError)).call(this, message));

        _this3.message = message;
        _this3.name = 'MapError';
        _this3.status = -1;
        return _this3;
    }

    return MapError;
}(Error);

var Galaxy = function () {

    // Constructor
    function Galaxy(plateifu, toggleon, redshift) {
        _classCallCheck(this, Galaxy);

        this.setPlateIfu(plateifu);
        this.toggleon = toggleon;
        this.redshift = redshift;
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
        this.maptab = $('#maptab');
        // toggle elements
        this.togglediv = $('#toggleinteract');
        this.toggleload = $('#toggle-load');
        this.togglediv.bootstrapToggle('off');
        this.toggleframe = $('#toggleframe');
        this.togglelines = $('#togglelines');
        // flag popover elements
        this.qualpop = $('#qualitypopover');
        this.targpops = $('.targpopovers');
        // maps elements
        this.dapmapsbut = $('#dapmapsbut');
        this.dapselect = $('#dapmapchoices');
        this.dapbt = $('#dapbtchoices');
        //this.dapselect.selectpicker('deselectAll');
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

        //Event Handlers
        this.maptab.on('click', this, this.resizeSpecView); // this event fires when a user clicks the MapSpec View Tab
        this.dapmapsbut.on('click', this, this.getDapMaps); // this event fires when a user clicks the GetMaps button
        this.resetmapsbut.on('click', this, this.resetMaps); // this event fires when a user clicks the Maps Reset button
        this.togglediv.on('change', this, this.initDynamic); // this event fires when a user clicks the Spec/Map View Toggle
        this.nsadisplay.on('click', this, this.displayNSA); // this event fires when a user clicks the NSA tab
        this.nsaresetbut.on('click', this, this.resetNSASelect); // this event fires when a user clicks the NSA select reset button
        //this.nsaselect.on('changed.bs.select', this, this.updateNSAPlot); // this event fires when a user selects an NSA parameter
        this.nsaplotbuttons.on('click', this, this.updateNSAPlot);
        //this.nsatable.on('page-change.bs.table', this, this.updateTableEvents);
        //this.nsatable.on('page-change.bs.table', this, this.updateTableEvents);
        this.toggleframe.on('change', this, this.toggleWavelength); // this event fires when a user clicks the Toggle Obs/Rest Frame
        this.togglelines.on('change', this, this.toggleLines); // this event fires when a user clicks the Toggle Lines
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

        // Resize the ouput MapSpec View when tab clicked

    }, {
        key: 'resizeSpecView',
        value: function resizeSpecView(event) {
            var _this = event.data;
            // wait 10 milliseconds before resizing so divs will have the correct size
            m.utils.window[0].setTimeout(function () {
                _this.webspec.resize();
                _this.olmap.map.updateSize();
            }, 10);
        }

        // Compute rest-frame wavelength

    }, {
        key: 'computeRestWave',
        value: function computeRestWave() {
            var _this4 = this;

            var rest = this.setSpectrumAxisFormatter('rest', this.redshift);
            this.obswave = this._spaxeldata.map(function (x) {
                return x[0];
            });
            this.restwave = this.obswave.map(function (x) {
                return rest(x);
            });
            this.rest_spaxeldata = this._spaxeldata.map(function (x, i) {
                return [_this4.restwave[i], x.slice(1)[0], x.slice(1)[1]];
            });
        }

        // Initialize and Load a DyGraph spectrum

    }, {
        key: 'loadSpaxel',
        value: function loadSpaxel(spaxel, title) {
            this._spaxeldata = spaxel;
            this.computeRestWave();
            // this plugin renables dygraphs 1.1 behaviour of unzooming to specified valueRange 
            var doubleClickZoomOutPlugin = {
                activate: function activate(g) {
                    // Save the initial y-axis range for later.
                    var initialValueRange = g.getOption('valueRange');
                    return {
                        dblclick: function dblclick(e) {
                            e.dygraph.updateOptions({
                                dateWindow: null, // zoom all the way out
                                valueRange: initialValueRange // zoom to a specific y-axis range.
                            });
                            e.preventDefault(); // prevent the default zoom out action.
                        }
                    };
                }
            };

            var labels = spaxel[0].length == 3 ? ['Wavelength', 'Flux', 'Model Fit'] : ['Wavelength', 'Flux'];
            var options = {
                title: title,
                labels: labels,
                legend: 'always',
                errorBars: true, // TODO DyGraph shows 2-sigma error bars FIX THIS
                ylabel: 'Flux [10<sup>-17</sup> erg/cm<sup>2</sup>/s/Å]',
                valueRange: [0, null],
                plugins: [doubleClickZoomOutPlugin]
            };
            var data = this.toggleframe.prop('checked') ? this.rest_spaxeldata : spaxel;
            options = this.addDygraphWaveOptions(options);
            this.webspec = new Dygraph(this.graphdiv[0], data, options);
        }

        // Dygraph Axis Formatter

    }, {
        key: 'setSpectrumAxisFormatter',
        value: function setSpectrumAxisFormatter(wave, redshift) {
            var obs = function obs(d, gran) {
                return d;
            };
            var rest = function rest(d, gran) {
                return parseFloat((d / (1 + redshift)).toPrecision(5));
            };

            if (wave === 'obs') {
                return obs;
            } else if (wave === 'rest') {
                return rest;
            }
        }
    }, {
        key: 'addDygraphWaveOptions',
        value: function addDygraphWaveOptions(oldoptions) {
            var newopts = {};
            if (this.toggleframe.prop('checked')) {
                newopts = { 'file': this.rest_spaxeldata, 'xlabel': 'Rest Wavelength [Ångströms]' };
            } else {
                newopts = { 'file': this._spaxeldata, 'xlabel': 'Observed Wavelength [Ångströms]' };
            }
            var options = Object.assign(oldoptions, newopts);
            return options;
        }

        // Toggle the Observed/Rest Wavelength

    }, {
        key: 'toggleWavelength',
        value: function toggleWavelength(event) {
            var _this = event.data;
            var options = {};
            options = _this.addDygraphWaveOptions(options);
            _this.webspec.updateOptions(options);
        }

        // Toggle Line Display

    }, {
        key: 'toggleLines',
        value: function toggleLines(event) {}

        // Update the spectrum message div for errors only

    }, {
        key: 'updateSpecMsg',
        value: function updateSpecMsg(specmsg, status) {
            this.specmsg.hide();
            if (status !== undefined && status === -1) {
                this.specmsg.show();
            }
            specmsg = specmsg.replace('<', '').replace('>', '');
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
            var mapchildren = this.mapsdiv.children('div');
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

        // Make Promise error message

    }, {
        key: 'makeError',
        value: function makeError(name) {
            return 'Unknown Error: the ' + name + ' javascript Ajax request failed!';
        }

        // Retrieves a new Spaxel from the server based on a given mouse position or xy spaxel coord.

    }, {
        key: 'getSpaxel',
        value: function getSpaxel(event) {
            var _this5 = this;

            var mousecoords = event.coordinate === undefined ? null : event.coordinate;
            var divid = $(event.target).parents('div').first().attr('id');
            var maptype = divid !== undefined && divid.search('highcharts') !== -1 ? 'heatmap' : 'optical';
            var x = event.point === undefined ? null : event.point.x;
            var y = event.point === undefined ? null : event.point.y;
            var keys = ['plateifu', 'image', 'imwidth', 'imheight', 'mousecoords', 'type', 'x', 'y'];
            var form = m.utils.buildForm(keys, this.plateifu, this.image, this.olmap.imwidth, this.olmap.imheight, mousecoords, maptype, x, y);

            // send the form data
            Promise.resolve($.post(Flask.url_for('galaxy_page.getspaxel'), form, 'json')).then(function (data) {
                if (data.result.status === -1) {
                    throw new SpaxelError('Error: ' + data.result.specmsg);
                }
                _this5.updateSpaxel(data.result.spectra, data.result.specmsg);
            }).catch(function (error) {
                var errmsg = error.message === undefined ? _this5.makeError('getSpaxel') : error.message;
                _this5.updateSpecMsg(errmsg, -1);
            });
        }

        // check the toggle preference on initial page load
        // eventually for user preferences

    }, {
        key: 'checkToggle',
        value: function checkToggle() {
            if (this.toggleon === 'true') {
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
            var _this6 = this;

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

                    // send the form data
                    Promise.resolve($.post(Flask.url_for('galaxy_page.initdynamic'), form, 'json')).then(function (data) {
                        if (data.result.error) {
                            var err = data.result.error;
                            throw new SpaxelError('Error : ' + err);
                        }
                        if (data.result.specstatus === -1) {
                            throw new SpaxelError('Error: ' + data.result.specmsg);
                        }
                        if (data.result.mapstatus === -1) {
                            throw new MapError('Error: ' + data.result.mapmsg);
                        }

                        var image = data.result.image;
                        var spaxel = data.result.spectra;
                        var spectitle = data.result.specmsg;
                        var maps = data.result.maps;
                        var mapmsg = data.result.mapmsg;
                        // Load the Galaxy Image
                        _this.initOpenLayers(image);
                        _this.toggleload.hide();

                        // Load the Spaxel and Maps
                        _this.loadSpaxel(spaxel, spectitle);
                        _this.initHeatmap(maps);
                        // refresh the map selectpicker
                        _this.dapselect.selectpicker('refresh');
                    }).catch(function (error) {
                        var errmsg = error.message === undefined ? _this6.makeError('initDynamic') : error.message;
                        _this.updateSpecMsg(errmsg, -1);
                        _this.updateMapMsg(errmsg, -1);
                    });
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

                var _popid$split = popid.split('_'),
                    _popid$split2 = _slicedToArray(_popid$split, 2),
                    base = _popid$split2[0],
                    targ = _popid$split2[1];
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
            Promise.resolve($.post(Flask.url_for('galaxy_page.updatemaps'), form, 'json')).then(function (data) {
                if (data.result.status === -1) {
                    throw new MapError('Error: ' + data.result.mapmsg);
                }
                _this.dapmapsbut.button('reset');
                _this.initHeatmap(data.result.maps);
            }).catch(function (error) {
                var errmsg = error.message === undefined ? _this.makeError('getDapMaps') : error.message;
                _this.updateMapMsg(errmsg, -1);
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
            mapmsg = mapmsg.replace('<', '').replace('>', '');
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
                Promise.resolve($.post(Flask.url_for('galaxy_page.initnsaplot'), form, 'json')).then(function (data) {
                    if (data.result.status !== 1) {
                        throw new Error('Error: ' + data.result.nsamsg);
                    }
                    _this.addNSAData(data.result.nsa);
                    _this.refreshNSASelect(data.result.nsachoices);
                    _this.initNSAScatter();
                    _this.setTableEvents();
                    _this.addNSAEvents();
                    _this.initNSABoxPlot(data.result.nsaplotcols);
                    _this.nsaload.hide();
                }).catch(function (error) {
                    var errmsg = error.message === undefined ? _this.makeError('displayNSA') : error.message;
                    _this.updateNSAMsg(errmsg, -1);
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
            var _this7 = this;

            var data = void 0,
                options = void 0;
            if (type === 'galaxy') {
                var x = this.mygalaxy[this.nsachoices[index].x];
                var y = this.mygalaxy[this.nsachoices[index].y];
                var pattern = 'absmag_[a-z]$';
                var xrev = this.nsachoices[index].x.search(pattern) > -1 ? true : false;
                var yrev = this.nsachoices[index].y.search(pattern) > -1 ? true : false;
                data = [{ 'name': this.plateifu, 'x': x, 'y': y }];
                options = { xtitle: this.nsachoices[index].xtitle, ytitle: this.nsachoices[index].ytitle,
                    title: this.nsachoices[index].title, galaxy: { name: this.plateifu }, xrev: xrev,
                    yrev: yrev };
            } else if (type === 'sample') {
                var _x = this.nsasample[this.nsachoices[index].x];
                var _y = this.nsasample[this.nsachoices[index].y];
                data = [];
                $.each(_x, function (index, value) {
                    if (value > -9999 && _y[index] > -9999) {
                        var tmp = { 'name': _this7.nsasample.plateifu[index], 'x': value, 'y': _y[index] };
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
            var _this8 = this;

            var tabledata = this.nsatable.bootstrapTable('getData');

            $.each(this.nsamovers, function (index, mover) {
                var id = mover.id;
                $('#' + id).on('dragstart', _this8, _this8.dragStart);
                $('#' + id).on('dragover', _this8, _this8.dragOver);
                $('#' + id).on('drop', _this8, _this8.moverDrop);
            });

            this.nsatable.on('page-change.bs.table', function () {
                $.each(tabledata, function (index, row) {
                    var mover = row[0];
                    var id = $(mover).attr('id');
                    $('#' + id).on('dragstart', _this8, _this8.dragStart);
                    $('#' + id).on('dragover', _this8, _this8.dragOver);
                    $('#' + id).on('drop', _this8, _this8.moverDrop);
                });
            });
        }

        // Add event handlers to the Highcharts scatter plots

    }, {
        key: 'addNSAEvents',
        value: function addNSAEvents() {
            var _this9 = this;

            //let _this = this;
            // NSA plot events
            this.nsaplots = $('.marvinplot');
            $.each(this.nsaplots, function (index, plot) {
                var id = plot.id;
                var highx = $('#' + id).find('.highcharts-xaxis');
                var highy = $('#' + id).find('.highcharts-yaxis');

                highx.on('dragover', _this9, _this9.dragOver);
                highx.on('dragenter', _this9, _this9.dragEnter);
                highx.on('drop', _this9, _this9.dropElement);
                highy.on('dragover', _this9, _this9.dragOver);
                highy.on('dragenter', _this9, _this9.dragEnter);
                highy.on('drop', _this9, _this9.dropElement);
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
            var _this10 = this;

            var data = [];
            this.nsaplotcols.forEach(function (column, index) {
                var goodsample = _this10.nsasample[column].filter(_this10.filterArray);
                var tmp = { 'value': _this10.mygalaxy[column], 'title': column, 'sample': goodsample };
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
            var data = void 0,
                options = void 0;
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
            var _this11 = this;

            // only update the single parent div element
            if (parentid !== undefined) {
                var parentdiv = this.maindiv.find('#' + parentid);
                var index = parseInt(parentid[parentid.length - 1]);

                var _updateNSAData = this.updateNSAData(index, 'galaxy'),
                    _updateNSAData2 = _slicedToArray(_updateNSAData, 2),
                    data = _updateNSAData2[0],
                    options = _updateNSAData2[1];

                var _updateNSAData3 = this.updateNSAData(index, 'sample'),
                    _updateNSAData4 = _slicedToArray(_updateNSAData3, 2),
                    sdata = _updateNSAData4[0],
                    soptions = _updateNSAData4[1];

                options.altseries = { data: sdata, name: 'Sample' };
                this.destroyChart(parentdiv, index);
                this.nsascatter[index] = new Scatter(parentdiv, data, options);
            } else {
                // try updating all of them
                this.nsascatter = {};
                $.each(this.nsaplots, function (index, plot) {
                    var plotdiv = $(plot);

                    var _updateNSAData5 = _this11.updateNSAData(index + 1, 'galaxy'),
                        _updateNSAData6 = _slicedToArray(_updateNSAData5, 2),
                        data = _updateNSAData6[0],
                        options = _updateNSAData6[1];

                    var _updateNSAData7 = _this11.updateNSAData(index + 1, 'sample'),
                        _updateNSAData8 = _slicedToArray(_updateNSAData7, 2),
                        sdata = _updateNSAData8[0],
                        soptions = _updateNSAData8[1];

                    options.altseries = { data: sdata, name: 'Sample' };
                    _this11.nsascatter[index + 1] = new Scatter(plotdiv, data, options);
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

            var _param$split = param.split('+'),
                _param$split2 = _slicedToArray(_param$split, 2),
                id = _param$split2[0],
                name = _param$split2[1];

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
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-26 21:47:05
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 13:09:41
*/

//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Header = function () {

    // Constructor
    function Header() {
        _classCallCheck(this, Header);

        this.navbar = $('.navbar');
        this.galidform = $('#headform');
        this.typeahead = $('#headform .typeahead');
        this.mplform = $('#mplform');
        this.mplselect = $('#mplselect');

        this.initTypeahead();

        //Event Handlers
        this.mplselect.on('change', this, this.selectMPL);
    }

    // Print


    _createClass(Header, [{
        key: 'print',
        value: function print() {
            console.log('I am Header!');
        }

        // Initialize the Typeahead

    }, {
        key: 'initTypeahead',
        value: function initTypeahead(typediv, formdiv, url, fxn) {

            var _this = this;
            typediv = typediv === undefined ? this.typeahead : $(typediv);
            formdiv = formdiv === undefined ? this.galidform : $(formdiv);
            var typeurl = url === undefined ? Flask.url_for('index_page.getgalidlist') : url;
            var afterfxn = fxn === undefined ? null : fxn;

            // create the bloodhound engine
            this.galids = new Bloodhound({
                datumTokenizer: Bloodhound.tokenizers.whitespace,
                queryTokenizer: Bloodhound.tokenizers.whitespace,
                //local:  ["(A)labama","Alaska","Arizona","Arkansas","Arkansas2","Barkansas", 'hello'],
                prefetch: typeurl,
                remote: {
                    url: typeurl,
                    filter: function filter(galids) {
                        return galids;
                    }
                }
            });

            // initialize the bloodhound suggestion engine
            this.galids.initialize();

            typediv.typeahead('destroy');
            typediv.typeahead({
                showHintOnFocus: true,
                items: 30,
                source: this.galids.ttAdapter(),
                afterSelect: function afterSelect() {
                    formdiv.submit();
                }
            });
        }

        // Select the MPL version on the web

    }, {
        key: 'selectMPL',
        value: function selectMPL(event) {
            var _this = event.data;
            var url = 'index_page.selectmpl';
            var verform = m.utils.serializeForm('#mplform');
            _this.sendAjax(verform, url, _this.reloadPage);
        }

        // Reload the Current Page

    }, {
        key: 'reloadPage',
        value: function reloadPage() {
            location.reload(true);
        }

        // Send an AJAX request

    }, {
        key: 'sendAjax',
        value: function sendAjax(form, url, fxn) {
            var _this = this;
            $.post(Flask.url_for(url), form, 'json').done(function (data) {
                // reload the current page, this re-instantiates a new Header with new version info from session
                if (data.result.status == 1) {
                    fxn();
                    _this.galids.clearPrefetchCache();
                    _this.galids.initialize();
                } else {
                    alert('Failed to set the versions! ' + data.result.msg);
                }
            }).fail(function (data) {
                alert('Failed to set the versions! Problem with Flask setversion. ' + data.result.msg);
            });
        }
    }]);

    return Header;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-08-30 11:28:26
 * @Last modified by:   andrews
 * @Last modified time: 2017-12-13 23:12:53
*/

//jshint esversion: 6
'use strict';

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var HeatMap = function () {

    // Constructor
    function HeatMap(mapdiv, data, title, plotparams, galthis) {
        _classCallCheck(this, HeatMap);

        if (data === undefined) {
            console.error('Must specify input map data to initialize a HeatMap!');
        } else if (mapdiv === undefined) {
            console.error('Must specify an input mapdiv to initialize a HeatMap');
        } else {
            this.mapdiv = mapdiv; // div element for map
            this.data = data; // map data
            this.title = title; // map title
            this.plotparams = plotparams; // default plotting parameters
            this.galthis = galthis; //the self of the Galaxy class
            this.parseTitle();
            this.initMap();
            this.setColorNoData(this, Highcharts);
        }
    }

    // test print


    _createClass(HeatMap, [{
        key: 'print',
        value: function print() {
            console.log('We are now printing heatmap for ', this.title);
        }

        // Parse the heatmap title into category, parameter, channel
        // e.g. 7443-1901: emline_gflux_ha-6564

    }, {
        key: 'parseTitle',
        value: function parseTitle() {
            var _title$split = this.title.split(':'),
                _title$split2 = _slicedToArray(_title$split, 2),
                plateifu = _title$split2[0],
                newtitle = _title$split2[1];

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
            var xylength = this.data.values.length;
            var xyrange = Array.apply(null, { length: xylength }).map(Number.call, Number);
            var zrange = [].concat.apply([], this.data.values);
            return [xyrange, zrange];
        }

        // Filter out null and no-data from z (DAP prop) data

    }, {
        key: 'filterRange',
        value: function filterRange(z) {
            if (z !== undefined && typeof z === 'number' && !isNaN(z)) {
                return true;
            } else {
                return false;
            }
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

            for (var ii = 0; ii < values.length; ii++) {
                for (var jj = 0; jj < values.length; jj++) {
                    var val = values[ii][jj];
                    var noData = void 0,
                        badData = void 0;
                    var signalToNoise = void 0,
                        signalToNoiseThreshold = void 0;

                    if (mask !== null) {
                        var bitmasks = this.plotparams["bits"];
                        noData = mask[ii][jj] & Math.pow(2, bitmasks["nocov"]);
                        badData = false;
                        for (var key in bitmasks["badData"]) {
                            badData = badData || mask[ii][jj] & Math.pow(2, bitmasks["badData"][key]);
                        }
                    } else {
                        noData = null;
                        badData = null;
                    }
                    signalToNoiseThreshold = this.plotparams["snr_min"];
                    if (ivar !== null) {
                        signalToNoise = Math.abs(val) * Math.sqrt(ivar[ii][jj]);
                    }

                    // value types
                    // val=no-data => gray color
                    // val=null => hatch area

                    if (noData) {
                        // for data that is outside the range "NOCOV" mask
                        val = 'no-data';
                    } else if (badData) {
                        // for data that is bad - masked in some way
                        val = null;
                    } else if (ivar !== null && signalToNoise < signalToNoiseThreshold) {
                        // for data that is low S/N
                        val = null;
                    } else if (ivar === null) {
                        // for data with no mask or no inverse variance extensions
                        if (this.title.search('binid') !== -1) {
                            // for binid extension only, set -1 values to no data
                            val = val == -1 ? 'no-data' : val;
                        } else if (val === 0.0) {
                            // set zero values to no-data
                            val = 'no-data';
                        }
                    }
                    // need to push as jj, ii since the numpy 2-d arrays are y, x based (row, col)
                    xyz.push([jj, ii, val]);
                }
            }
            return xyz;
        }
    }, {
        key: 'setColorNoData',
        value: function setColorNoData(_this, H) {
            H.wrap(H.ColorAxis.prototype, 'toColor', function (proceed, value, point) {
                if (value === 'no-data') {
                    // make gray color
                    return 'rgba(0,0,0,0)'; // '#A8A8A8';
                } else if (value === 'low-sn') {
                    // make light blue with half-opacity == muddy blue-gray
                    return 'rgba(0,191,255,0.5)'; //'#7fffd4';
                } else return proceed.apply(this, Array.prototype.slice.call(arguments, 1));
            });
        }
    }, {
        key: 'setColorMapHex',
        value: function setColorMapHex(cmap) {

            var linearLabHex = ['#040404', '#0a0308', '#0d040b', '#10050e', '#120510', '#150612', '#160713', '#180815', '#1a0816', '#1b0918', '#1c0a19', '#1e0b1a', '#1f0c1b', '#200c1c', '#210d1d', '#230e1f', '#240e20', '#250f20', '#260f21', '#271022', '#281123', '#291124', '#2a1226', '#2b1326', '#2c1327', '#2e1429', '#2e142d', '#2e1532', '#2d1537', '#2d153c', '#2d1640', '#2d1743', '#2d1747', '#2d184b', '#2d184d', '#2d1951', '#2d1954', '#2c1a57', '#2c1b5a', '#2d1b5c', '#2d1c5f', '#2c1d62', '#2c1d64', '#2c1e67', '#2c1f6a', '#2c1f6d', '#2c206e', '#2c2171', '#2c2274', '#2c2276', '#2a2379', '#282678', '#262877', '#242a78', '#222c78', '#212e78', '#202f78', '#1f3179', '#1e327a', '#1e337b', '#1d347b', '#1d357d', '#1c377d', '#1c387e', '#1b397f', '#1c3a80', '#1c3b81', '#1b3c81', '#1b3d83', '#1b3e84', '#1b3f85', '#1c4086', '#1b4187', '#1b4288', '#1b4489', '#1b458a', '#194788', '#164986', '#154a85', '#144c83', '#114e81', '#104f80', '#0f517e', '#0e527d', '#0a547b', '#0a557a', '#095778', '#085877', '#075976', '#065b75', '#045c73', '#045e72', '#045f72', '#036070', '#01626f', '#01636e', '#00646d', '#00656c', '#00676b', '#00686a', '#006969', '#006b68', '#006c65', '#006e64', '#006f63', '#007062', '#007260', '#00735f', '#00745d', '#00765c', '#00775a', '#007859', '#007958', '#007b56', '#007c55', '#007d53', '#007f52', '#008050', '#00814f', '#00834d', '#00844b', '#008549', '#008648', '#008846', '#008944', '#008a42', '#008b41', '#008d40', '#008e3f', '#008f3d', '#00913c', '#00923c', '#00933a', '#009539', '#009638', '#009737', '#009935', '#009a34', '#009b33', '#009d32', '#009e30', '#009f2f', '#00a02d', '#00a22c', '#00a32a', '#00a429', '#00a527', '#00a724', '#00a822', '#00a91f', '#00aa17', '#00a908', '#09aa00', '#14ab00', '#1dac00', '#23ad00', '#28ae00', '#2daf00', '#30b000', '#34b100', '#37b200', '#3bb300', '#3db400', '#40b500', '#42b600', '#44b700', '#47b800', '#49b900', '#4cba00', '#4ebb00', '#4fbc00', '#51bd00', '#53be00', '#55bf00', '#57c000', '#5cc000', '#63c100', '#6ac100', '#72c100', '#77c200', '#7dc200', '#82c200', '#87c300', '#8cc300', '#91c300', '#95c400', '#99c400', '#9dc500', '#a1c500', '#a5c500', '#a9c600', '#acc600', '#b0c700', '#b4c700', '#b8c700', '#bac800', '#bec900', '#c1c900', '#c5c900', '#c8ca00', '#c9c918', '#cbca33', '#ceca41', '#cfcb4d', '#d1cb57', '#d4cb5f', '#d5cc67', '#d7cd6d', '#dacd74', '#dbce79', '#ddcf7f', '#dfcf84', '#e2cf8a', '#e3d08f', '#e5d193', '#e7d197', '#e8d29b', '#ebd39f', '#edd3a4', '#eed4a8', '#f0d4ac', '#f3d5af', '#f3d6b3', '#f5d6b7', '#f8d7ba', '#f8d8bd', '#f8dac1', '#f7dbc3', '#f7dcc6', '#f7dec9', '#f8dfcc', '#f7e0ce', '#f7e2d1', '#f7e3d3', '#f7e5d6', '#f7e6d8', '#f7e7da', '#f7e8dc', '#f8eae0', '#f7ebe1', '#f7ece5', '#f7eee7', '#f7efe8', '#f8f0eb', '#f8f2ed', '#f7f3ef', '#f8f4f1', '#f8f6f4', '#f8f7f6', '#f8f8f8', '#f9f9f9', '#fbfbfb', '#fcfcfc', '#fdfdfd', '#fefefe', '#ffffff'];

            var infernoHex = ['#000004', '#010005', '#010106', '#010108', '#02010a', '#02020c', '#02020e', '#030210', '#040312', '#040314', '#050417', '#060419', '#07051b', '#08051d', '#09061f', '#0a0722', '#0b0724', '#0c0826', '#0d0829', '#0e092b', '#10092d', '#110a30', '#120a32', '#140b34', '#150b37', '#160b39', '#180c3c', '#190c3e', '#1b0c41', '#1c0c43', '#1e0c45', '#1f0c48', '#210c4a', '#230c4c', '#240c4f', '#260c51', '#280b53', '#290b55', '#2b0b57', '#2d0b59', '#2f0a5b', '#310a5c', '#320a5e', '#340a5f', '#360961', '#380962', '#390963', '#3b0964', '#3d0965', '#3e0966', '#400a67', '#420a68', '#440a68', '#450a69', '#470b6a', '#490b6a', '#4a0c6b', '#4c0c6b', '#4d0d6c', '#4f0d6c', '#510e6c', '#520e6d', '#540f6d', '#550f6d', '#57106e', '#59106e', '#5a116e', '#5c126e', '#5d126e', '#5f136e', '#61136e', '#62146e', '#64156e', '#65156e', '#67166e', '#69166e', '#6a176e', '#6c186e', '#6d186e', '#6f196e', '#71196e', '#721a6e', '#741a6e', '#751b6e', '#771c6d', '#781c6d', '#7a1d6d', '#7c1d6d', '#7d1e6d', '#7f1e6c', '#801f6c', '#82206c', '#84206b', '#85216b', '#87216b', '#88226a', '#8a226a', '#8c2369', '#8d2369', '#8f2469', '#902568', '#922568', '#932667', '#952667', '#972766', '#982766', '#9a2865', '#9b2964', '#9d2964', '#9f2a63', '#a02a63', '#a22b62', '#a32c61', '#a52c60', '#a62d60', '#a82e5f', '#a92e5e', '#ab2f5e', '#ad305d', '#ae305c', '#b0315b', '#b1325a', '#b3325a', '#b43359', '#b63458', '#b73557', '#b93556', '#ba3655', '#bc3754', '#bd3853', '#bf3952', '#c03a51', '#c13a50', '#c33b4f', '#c43c4e', '#c63d4d', '#c73e4c', '#c83f4b', '#ca404a', '#cb4149', '#cc4248', '#ce4347', '#cf4446', '#d04545', '#d24644', '#d34743', '#d44842', '#d54a41', '#d74b3f', '#d84c3e', '#d94d3d', '#da4e3c', '#db503b', '#dd513a', '#de5238', '#df5337', '#e05536', '#e15635', '#e25734', '#e35933', '#e45a31', '#e55c30', '#e65d2f', '#e75e2e', '#e8602d', '#e9612b', '#ea632a', '#eb6429', '#eb6628', '#ec6726', '#ed6925', '#ee6a24', '#ef6c23', '#ef6e21', '#f06f20', '#f1711f', '#f1731d', '#f2741c', '#f3761b', '#f37819', '#f47918', '#f57b17', '#f57d15', '#f67e14', '#f68013', '#f78212', '#f78410', '#f8850f', '#f8870e', '#f8890c', '#f98b0b', '#f98c0a', '#f98e09', '#fa9008', '#fa9207', '#fa9407', '#fb9606', '#fb9706', '#fb9906', '#fb9b06', '#fb9d07', '#fc9f07', '#fca108', '#fca309', '#fca50a', '#fca60c', '#fca80d', '#fcaa0f', '#fcac11', '#fcae12', '#fcb014', '#fcb216', '#fcb418', '#fbb61a', '#fbb81d', '#fbba1f', '#fbbc21', '#fbbe23', '#fac026', '#fac228', '#fac42a', '#fac62d', '#f9c72f', '#f9c932', '#f9cb35', '#f8cd37', '#f8cf3a', '#f7d13d', '#f7d340', '#f6d543', '#f6d746', '#f5d949', '#f5db4c', '#f4dd4f', '#f4df53', '#f4e156', '#f3e35a', '#f3e55d', '#f2e661', '#f2e865', '#f2ea69', '#f1ec6d', '#f1ed71', '#f1ef75', '#f1f179', '#f2f27d', '#f2f482', '#f3f586', '#f3f68a', '#f4f88e', '#f5f992', '#f6fa96', '#f8fb9a', '#f9fc9d', '#fafda1', '#fcffa4'];

            var RdBuHex = ['#053061', '#063264', '#073467', '#08366a', '#09386d', '#0a3b70', '#0c3d73', '#0d3f76', '#0e4179', '#0f437b', '#10457e', '#114781', '#124984', '#134c87', '#144e8a', '#15508d', '#175290', '#185493', '#195696', '#1a5899', '#1b5a9c', '#1c5c9f', '#1d5fa2', '#1e61a5', '#1f63a8', '#2065ab', '#2267ac', '#2369ad', '#246aae', '#266caf', '#276eb0', '#2870b1', '#2a71b2', '#2b73b3', '#2c75b4', '#2e77b5', '#2f79b5', '#307ab6', '#327cb7', '#337eb8', '#3480b9', '#3681ba', '#3783bb', '#3885bc', '#3a87bd', '#3b88be', '#3c8abe', '#3e8cbf', '#3f8ec0', '#408fc1', '#4291c2', '#4393c3', '#4695c4', '#4997c5', '#4c99c6', '#4f9bc7', '#529dc8', '#569fc9', '#59a1ca', '#5ca3cb', '#5fa5cd', '#62a7ce', '#65a9cf', '#68abd0', '#6bacd1', '#6eaed2', '#71b0d3', '#75b2d4', '#78b4d5', '#7bb6d6', '#7eb8d7', '#81bad8', '#84bcd9', '#87beda', '#8ac0db', '#8dc2dc', '#90c4dd', '#93c6de', '#96c7df', '#98c8e0', '#9bc9e0', '#9dcbe1', '#a0cce2', '#a2cde3', '#a5cee3', '#a7d0e4', '#a9d1e5', '#acd2e5', '#aed3e6', '#b1d5e7', '#b3d6e8', '#b6d7e8', '#b8d8e9', '#bbdaea', '#bddbea', '#c0dceb', '#c2ddec', '#c5dfec', '#c7e0ed', '#cae1ee', '#cce2ef', '#cfe4ef', '#d1e5f0', '#d2e6f0', '#d4e6f1', '#d5e7f1', '#d7e8f1', '#d8e9f1', '#dae9f2', '#dbeaf2', '#ddebf2', '#deebf2', '#e0ecf3', '#e1edf3', '#e3edf3', '#e4eef4', '#e6eff4', '#e7f0f4', '#e9f0f4', '#eaf1f5', '#ecf2f5', '#edf2f5', '#eff3f5', '#f0f4f6', '#f2f5f6', '#f3f5f6', '#f5f6f7', '#f6f7f7', '#f7f6f6', '#f7f5f4', '#f8f4f2', '#f8f3f0', '#f8f2ef', '#f8f1ed', '#f9f0eb', '#f9efe9', '#f9eee7', '#f9ede5', '#f9ebe3', '#faeae1', '#fae9df', '#fae8de', '#fae7dc', '#fbe6da', '#fbe5d8', '#fbe4d6', '#fbe3d4', '#fce2d2', '#fce0d0', '#fcdfcf', '#fcdecd', '#fdddcb', '#fddcc9', '#fddbc7', '#fdd9c4', '#fcd7c2', '#fcd5bf', '#fcd3bc', '#fbd0b9', '#fbceb7', '#fbccb4', '#facab1', '#fac8af', '#f9c6ac', '#f9c4a9', '#f9c2a7', '#f8bfa4', '#f8bda1', '#f8bb9e', '#f7b99c', '#f7b799', '#f7b596', '#f6b394', '#f6b191', '#f6af8e', '#f5ac8b', '#f5aa89', '#f5a886', '#f4a683', '#f3a481', '#f2a17f', '#f19e7d', '#f09c7b', '#ef9979', '#ee9677', '#ec9374', '#eb9172', '#ea8e70', '#e98b6e', '#e8896c', '#e6866a', '#e58368', '#e48066', '#e37e64', '#e27b62', '#e17860', '#df765e', '#de735c', '#dd7059', '#dc6e57', '#db6b55', '#da6853', '#d86551', '#d7634f', '#d6604d', '#d55d4c', '#d35a4a', '#d25849', '#d05548', '#cf5246', '#ce4f45', '#cc4c44', '#cb4942', '#c94741', '#c84440', '#c6413e', '#c53e3d', '#c43b3c', '#c2383a', '#c13639', '#bf3338', '#be3036', '#bd2d35', '#bb2a34', '#ba2832', '#b82531', '#b72230', '#b61f2e', '#b41c2d', '#b3192c', '#b1182b', '#ae172a', '#ab162a', '#a81529', '#a51429', '#a21328', '#9f1228', '#9c1127', '#991027', '#960f27', '#930e26', '#900d26', '#8d0c25', '#8a0b25', '#870a24', '#840924', '#810823', '#7f0823', '#7c0722', '#790622', '#760521', '#730421', '#700320', '#6d0220', '#6a011f', '#67001f'];

            if (cmap === "linearlab") {
                return linearLabHex;
            } else if (cmap === "inferno") {
                return infernoHex;
            } else if (cmap === "RdBu_r") {
                return RdBuHex;
            } else {
                return ["#000000", "#FFFFFF"];
            }
        }
    }, {
        key: 'setColorStops',
        value: function setColorStops(cmap) {
            var colorHex = this.setColorMapHex(cmap);
            var stopLocations = colorHex.length;
            var colormap = new Array(stopLocations);
            for (var ii = 0; ii < stopLocations; ii++) {
                colormap[ii] = [ii / (stopLocations - 1), colorHex[ii]];
            }
            return colormap;
        }
    }, {
        key: 'quantileClip',
        value: function quantileClip(range) {
            var quantLow = void 0,
                quantHigh = void 0,
                zQuantLow = void 0,
                zQuantHigh = void 0;

            var _plotparams$percentil = _slicedToArray(this.plotparams["percentile_clip"], 2);

            quantLow = _plotparams$percentil[0];
            quantHigh = _plotparams$percentil[1];

            var _getMinMax = this.getMinMax(range);

            var _getMinMax2 = _slicedToArray(_getMinMax, 2);

            zQuantLow = _getMinMax2[0];
            zQuantHigh = _getMinMax2[1];

            if (range.length > 0) {
                if (quantLow > 0) {
                    zQuantLow = math.quantileSeq(range, quantLow / 100);
                }
                if (quantHigh < 100) {
                    zQuantHigh = math.quantileSeq(range, quantHigh / 100);
                }
            }
            return [zQuantLow, zQuantHigh];
        }

        // initialize the heat map

    }, {
        key: 'initMap',
        value: function initMap() {
            // set the galaxy class self to a variable
            var _galthis = this.galthis;

            // get the ranges
            //let range  = this.getXRange();
            var xyrange = void 0,
                zrange = void 0;

            // get the min and max of the ranges
            var _getRange = this.getRange();

            var _getRange2 = _slicedToArray(_getRange, 2);

            xyrange = _getRange2[0];
            zrange = _getRange2[1];
            var xymin = void 0,
                xymax = void 0,
                zmin = void 0,
                zmax = void 0;

            var _getMinMax3 = this.getMinMax(xyrange);

            var _getMinMax4 = _slicedToArray(_getMinMax3, 2);

            xymin = _getMinMax4[0];
            xymax = _getMinMax4[1];

            // set null data and create new zrange, min, and max
            var _getMinMax5 = this.getMinMax(zrange);

            var _getMinMax6 = _slicedToArray(_getMinMax5, 2);

            zmin = _getMinMax6[0];
            zmax = _getMinMax6[1];
            var data = this.setNull(this.data);
            zrange = data.map(function (o) {
                return o[2];
            });
            zrange = zrange.filter(this.filterRange);
            // [zmin, zmax] = this.getMinMax(zrange);

            var _quantileClip = this.quantileClip(zrange);

            var _quantileClip2 = _slicedToArray(_quantileClip, 2);

            zmin = _quantileClip2[0];
            zmax = _quantileClip2[1];


            var cmap = this.plotparams["cmap"];

            // make color bar symmetric
            if (this.plotparams["symmetric"]) {
                var zabsmax = Math.max.apply(null, [Math.abs(zmin), Math.abs(zmax)]);
                zmin = -zabsmax;
                zmax = zabsmax;
            }

            var cstops = this.setColorStops(cmap);

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
                credits: { enabled: false },
                title: {
                    text: this.title.replace(/[_]/g, " "),
                    style: { fontSize: "14px" }
                },
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
                    min: zmin,
                    max: zmax,
                    minColor: cstops[0][1],
                    maxColor: cstops[cstops.length - 1][1],
                    stops: cstops,
                    labels: { align: 'center' },
                    reversed: false,
                    startOnTick: false,
                    endOnTick: false,
                    tickPixelInterval: 30,
                    type: "linear"
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
                            // fill: 'rgba(255, 255, 255, 1)'  // 'rgba(168, 168, 168, 0.3)'
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
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 11:24:07
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-28 10:24:41
*/

//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Marvin = function () {
    function Marvin(options) {
        _classCallCheck(this, Marvin);

        // set options
        //_.defaults(options, {fruit: "strawberry"})
        this.options = options;

        // set up utility functions
        this.utils = new Utils();
        this.utils.print();
        this.utils.initInfoPopOvers();
        this.utils.initToolTips();

        // load the header
        this.header = new Header();
        this.header.print();

        // setup raven
        this.setupRaven();

        // check the browser on page load
        this.window = $(window);
        // this.window.on('load', this.checkBrowser);
    }

    // sets the Sentry raven for monitoring


    _createClass(Marvin, [{
        key: 'setupRaven',
        value: function setupRaven() {
            Raven.config('https://98bc7162624049ffa3d8d9911e373430@sentry.io/107924', {
                release: '0.2.0b1',
                // we highly recommend restricting exceptions to a domain in order to filter out clutter
                whitelistUrls: ['/(sas|api)\.sdss\.org/marvin/', '/(sas|api)\.sdss\.org/marvin2/'],
                includePaths: ['/https?:\/\/((sas|api)\.)?sdss\.org/marvin', '/https?:\/\/((sas|api)\.)?sdss\.org/marvin2']
            }).install();
        }

        // check the browser for banner display

    }, {
        key: 'checkBrowser',
        value: function checkBrowser(event) {
            var _this = event.data;
            if (!!navigator.userAgent.match(/Version\/[\d\.]+.*Safari/)) {
                m.utils.marvinBanner('We have detected that you are using Safari. Some features may not work as expected. We recommend using Chrome or Firefox.', 1, 'safari_banner', 'http://sdss-marvin.readthedocs.io/en/latest/known-issues.html#known-browser');
            }
        }
    }]);

    return Marvin;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 17:38:25
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 01:43:36
*/

//
// Javascript object handling all things related to OpenLayers Map
//
//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var OLMap = function () {

    // Constructor
    function OLMap(image) {
        _classCallCheck(this, OLMap);

        if (image === undefined) {
            console.error('Must specify an input image to initialize a Map!');
        } else {
            this.image = image;
            this.staticimdiv = $('#staticimage')[0];
            this.mapdiv = $('#imagediv')[0];
            this.getImageSize();
            this.setProjection();
            this.setView();
            this.initMap();
            this.addDrawInteraction();
        }
    }

    // test print


    _createClass(OLMap, [{
        key: 'print',
        value: function print() {
            console.log('We are now printing openlayers map');
        }

        // Get the natural size of the input static image

    }, {
        key: 'getImageSize',
        value: function getImageSize() {
            if (this.staticimdiv !== undefined) {
                this.imwidth = this.staticimdiv.naturalWidth;
                this.imheight = this.staticimdiv.naturalHeight;
            }
        }

        // Set the mouse position control

    }, {
        key: 'setMouseControl',
        value: function setMouseControl() {
            var mousePositionControl = new ol.control.MousePosition({
                coordinateFormat: ol.coordinate.createStringXY(4),
                projection: 'EPSG:4326',
                // comment the following two lines to have the mouse position be placed within the map.
                //className: 'custom-mouse-position',
                //target: document.getElementById('mouse-position'),
                undefinedHTML: '&nbsp;'
            });
            return mousePositionControl;
        }

        // Set the image Projection

    }, {
        key: 'setProjection',
        value: function setProjection() {
            this.extent = [0, 0, this.imwidth, this.imheight];
            this.projection = new ol.proj.Projection({
                code: 'ifu',
                units: 'pixels',
                extent: this.extent
            });
        }

        // Set the base image Layer

    }, {
        key: 'setBaseImageLayer',
        value: function setBaseImageLayer() {
            var imagelayer = new ol.layer.Image({
                source: new ol.source.ImageStatic({
                    url: this.image,
                    projection: this.projection,
                    imageExtent: this.extent
                })
            });
            return imagelayer;
        }

        // Set the image View

    }, {
        key: 'setView',
        value: function setView() {
            this.view = new ol.View({
                projection: this.projection,
                center: ol.extent.getCenter(this.extent),
                zoom: 1,
                maxZoom: 8,
                maxResolution: 1.4
            });
        }

        // Initialize the Map

    }, {
        key: 'initMap',
        value: function initMap() {
            var mousePositionControl = this.setMouseControl();
            var baseimage = this.setBaseImageLayer();
            this.map = new ol.Map({
                controls: ol.control.defaults({
                    attributionOptions: /** @type {olx.control.AttributionOptions} */{
                        collapsible: false
                    }
                }).extend([mousePositionControl]),
                layers: [baseimage],
                target: this.mapdiv,
                view: this.view
            });
        }

        // Add a Draw Interaction

    }, {
        key: 'addDrawInteraction',
        value: function addDrawInteraction() {
            // set up variable for last saved feature & vector source for point
            var lastFeature = void 0;
            var drawsource = new ol.source.Vector({ wrapX: false });
            // create new point vectorLayer
            var pointVector = this.newVectorLayer(drawsource);
            // add the layer to the map
            this.map.addLayer(pointVector);

            // New draw event ; default to Point
            var value = 'Point';
            var geometryFunction = void 0,
                maxPoints = void 0;
            this.draw = new ol.interaction.Draw({
                source: drawsource,
                type: /** @type {ol.geom.GeometryType} */value,
                geometryFunction: geometryFunction,
                maxPoints: maxPoints
            });

            // On draw end, remove the last saved feature (point)
            this.draw.on('drawend', function (e) {
                if (lastFeature) {
                    drawsource.removeFeature(lastFeature);
                }
                lastFeature = e.feature;
            });

            // add draw interaction onto the map
            this.map.addInteraction(this.draw);
        }

        // New Vector Layer

    }, {
        key: 'newVectorLayer',
        value: function newVectorLayer(source) {
            // default set to Point, but eventually expand this to different vector layer types
            var vector = new ol.layer.Vector({
                source: source,
                style: new ol.style.Style({
                    fill: new ol.style.Fill({
                        color: 'rgba(255, 255, 255, 0.2)'
                    }),
                    stroke: new ol.style.Stroke({
                        color: '#FF0808',
                        width: 2
                    }),
                    image: new ol.style.Circle({
                        radius: 3,
                        fill: new ol.style.Fill({
                            color: '#FF0808'
                        })
                    })
                })
            });
            return vector;
        }
    }]);

    return OLMap;
}();
;/*
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
;/*
* @Author: Brian Cherinka
* @Date:   2016-05-13 13:26:21
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-06-28 11:59:04
*/

//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Search = function () {

    // Constructor
    function Search() {
        _classCallCheck(this, Search);

        this.searchform = $('#searchform');
        this.typeahead = $('#searchform .typeahead');
        this.returnparams = $('#returnparams');
        this.parambox = $('#parambox');
        this.searchbox = $("#searchbox");

        this.builder = $('#builder');
        this.sqlalert = $('#sqlalert');
        this.getsql = $('#get-sql');
        this.resetsql = $('#reset-sql');
        this.runsql = $('#run-sql');

        // Event Handlers
        this.getsql.on('click', this, this.getSQL);
        this.resetsql.on('click', this, this.resetSQL);
        this.runsql.on('click', this, this.runSQL);
    }

    // Print


    _createClass(Search, [{
        key: 'print',
        value: function print() {
            console.log('I am Search!');
        }

        // Extract

    }, {
        key: 'extractor',
        value: function extractor(input) {
            var regexp = new RegExp('([^,]+)$');
            // parse input for newly typed text
            var result = regexp.exec(input);
            // select last entry after comma
            if (result && result[1]) {
                return result[1].trim();
            }
            return '';
        }

        // Initialize Query Param Typeahead

    }, {
        key: 'initTypeahead',
        value: function initTypeahead(typediv, formdiv, url, fxn) {

            var _this = this;
            var typeurl = void 0;
            typediv = typediv === undefined ? this.typeahead : $(typediv);
            formdiv = formdiv === undefined ? this.searchform : $(formdiv);
            // get the typeahead search page getparams url
            try {
                typeurl = url === undefined ? Flask.url_for('search_page.getparams', { 'paramdisplay': 'best' }) : url;
            } catch (error) {
                Raven.captureException(error);
                console.error('Error getting search getparams url:', error);
            }
            var afterfxn = fxn === undefined ? null : fxn;

            function customQueryTokenizer(str) {
                var newstr = str.toString();
                return [_this.extractor(newstr)];
            };

            // create the bloodhound engine
            this.queryparams = new Bloodhound({
                datumTokenizer: Bloodhound.tokenizers.whitespace,
                //queryTokenizer: Bloodhound.tokenizers.whitespace,
                queryTokenizer: customQueryTokenizer,
                prefetch: typeurl,
                remote: {
                    url: typeurl,
                    filter: function filter(qpars) {
                        return qpars;
                    }
                }
            });

            // initialize the bloodhound suggestion engine
            this.queryparams.initialize();

            // init the search typeahead
            typediv.typeahead('destroy');
            typediv.typeahead({
                showHintOnFocus: true,
                items: 'all',
                source: this.queryparams.ttAdapter(),
                updater: function updater(item) {
                    // used to updated the input box with selected option
                    // item = selected item from dropdown
                    var currenttext = this.$element.val();
                    var removedtemptype = currenttext.replace(/[^,]*$/, '');
                    var newtext = removedtemptype + item + ', ';
                    return newtext;
                },
                matcher: function matcher(item) {
                    // used to determined if a query matches an item
                    var tquery = _this.extractor(this.query);
                    console.log('query', this.query);
                    console.log(tquery);
                    if (!tquery) return false;
                    return ~item.toLowerCase().indexOf(tquery.toLowerCase());
                },
                highlighter: function highlighter(item) {
                    // used to highlight autocomplete results ; returns html
                    var oquery = _this.extractor(this.query);
                    var query = oquery.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, '\\$&');
                    return item.replace(new RegExp('(' + query + ')', 'ig'), function ($1, match) {
                        return '<strong>' + match + '</strong>';
                    });
                }
            });
        }

        // Setup Query Builder

    }, {
        key: 'setupQB',
        value: function setupQB(params) {
            $('.modal-dialog').draggable(); // makes the modal dialog draggable
            this.builder.queryBuilder({ plugins: ['bt-selectpicker', 'not-group', 'invert'], filters: params,
                operators: ['equal', 'not_equal', 'less', 'less_or_equal', 'greater', 'greater_or_equal', 'between', 'contains', 'begins_with', 'ends_with'] });
        }

        // Get the SQL from the QB

    }, {
        key: 'getSQL',
        value: function getSQL(event) {
            var _this = event.data;
            try {
                var result = _this.builder.queryBuilder('getSQL', false);
                if (result.sql.length) {
                    _this.sqlalert.html("");
                    // remove the quotations
                    var newsql = result.sql.replace(/[']+/g, "");
                    // replace any like and percents with = and *
                    var likeidx = newsql.indexOf('LIKE');
                    if (likeidx !== -1) {
                        newsql = newsql.replace('LIKE(', '= ').replace(/[%]/g, '*');
                        var idx = newsql.indexOf(')', likeidx);
                        newsql = newsql.replace(newsql.charAt(idx), " ");
                    }
                    _this.searchbox.val(newsql);
                }
            } catch (error) {
                _this.sqlalert.html("<p class='text-center text-danger'>Must provide valid input.</p>");
            }
        }

        // Reset the SQL in SearchBox

    }, {
        key: 'resetSQL',
        value: function resetSQL(event) {
            var _this = event.data;
            _this.searchbox.val("");
        }

        // Run the Query from the QB

    }, {
        key: 'runSQL',
        value: function runSQL(event) {
            var _this = event.data;
            if (_this.searchbox.val() === "") {
                _this.sqlalert.html("<p class='text-center text-danger'>You must generate SQL first!</p>");
            } else {
                _this.searchform.submit();
            }
        }
    }]);

    return Search;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-25 13:56:19
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-09-28 13:25:11
*/

//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Table = function () {

    // Constructor
    function Table(tablediv) {
        _classCallCheck(this, Table);

        this.setTable(tablediv);

        // Event Handlers
        this.table.on('load-success.bs.table', this, this.setSuccessMsg);
        this.table.on('load-error.bs.table', this, this.setErrMsg);
    }

    // Print


    _createClass(Table, [{
        key: 'print',
        value: function print() {
            console.log('I am Table!');
        }

        // Set the initial Table

    }, {
        key: 'setTable',
        value: function setTable(tablediv) {
            if (tablediv !== undefined) {
                console.log('setting the table');
                this.table = tablediv;
                this.errdiv = this.table.siblings('#errdiv');
                this.tableerr = this.errdiv.find('#tableerror');
                this.tableerr.hide();
            }
        }

        // initialize a table

    }, {
        key: 'initTable',
        value: function initTable(url, data) {
            this.url = url;
            var cols = void 0;

            // if data
            if (data.columns !== null) {
                cols = this.makeColumns(data.columns);
            }

            console.log(data);
            console.log('cols', cols);
            // init the Bootstrap table
            this.table.bootstrapTable({
                classes: 'table table-bordered table-condensed table-hover',
                toggle: 'table',
                toolbar: '#toolbar',
                pagination: true,
                pageSize: 10,
                pageList: '[10, 20, 50]',
                sidePagination: 'server',
                method: 'post',
                contentType: "application/x-www-form-urlencoded",
                data: data.rows,
                totalRows: data.total,
                columns: cols,
                url: url,
                showColumns: true,
                showToggle: true,
                sortName: 'mangaid',
                sortOrder: 'asc',
                formatNoMatches: function formatNoMatches() {
                    return "This table is empty...";
                }
            });
        }

        // update the error div with a message

    }, {
        key: 'updateMsg',
        value: function updateMsg(msg) {
            var errmsg = '<strong>' + msg + '</strong>';
            this.tableerr.html(errmsg);
            this.tableerr.show();
        }

        // set a table error message

    }, {
        key: 'setErrMsg',
        value: function setErrMsg(event, status, res) {
            var _this = event.data;
            var extra = '';
            if (status === 502) {
                extra = 'bad server response retrieving web table.  likely uncaught error on server side.  check logs.';
            }
            var msg = 'Status ' + status + ' - ' + res.statusText + ': ' + extra;
            _this.updateMsg(msg);
        }

        // set a table error message

    }, {
        key: 'setSuccessMsg',
        value: function setSuccessMsg(event, data) {
            var _this = event.data;
            _this.tableerr.hide();
            if (data.status === -1) {
                _this.updateMsg(data.errmsg);
            }
        }

        // make the Table Columns

    }, {
        key: 'makeColumns',
        value: function makeColumns(columns) {
            var _this2 = this;

            var cols = [];
            columns.forEach(function (name, index) {
                var colmap = {};
                colmap.field = name;
                colmap.title = name;
                colmap.sortable = true;
                if (name.match('plateifu|mangaid')) {
                    colmap.formatter = _this2.linkformatter;
                }
                cols.push(colmap);
            });
            return cols;
        }

        // Link Formatter

    }, {
        key: 'linkformatter',
        value: function linkformatter(value, row, index) {
            var url = Flask.url_for('galaxy_page.Galaxy:get', { 'galid': value });
            var link = '<a href=' + url + ' target=\'_blank\'>' + value + '</a>';
            return link;
        }

        // Handle the Bootstrap table JSON response

    }, {
        key: 'handleResponse',
        value: function handleResponse(results) {
            // load the bootstrap table div
            if (this.table === null) {
                this.setTable();
            }
            this.table = $('#table');
            // Get new columns
            var cols = results.columns;
            cols = [];
            results.columns.forEach(function (name, index) {
                var colmap = {};
                colmap.field = name;
                colmap.title = name;
                colmap.sortable = true;
                cols.push(colmap);
            });

            // Load new options
            this.table.bootstrapTable('refreshOptions', { 'columns': cols, 'totalRows': results.total });

            return results;
        }
    }]);

    return Table;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-12 00:10:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2018-04-13 16:05:17
*/

// Javascript code for general things
//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Utils = function () {

    // Constructor
    function Utils() {
        _classCallCheck(this, Utils);

        this.window = $(window);

        // login handlers
        $('#login-user').on('keyup', this, this.submitLogin); // submit login on keypress
        $('#login-pass').on('keyup', this, this.submitLogin); // submit login on keypress
        $('#login-drop').on('hide.bs.dropdown', this, this.resetLogin); //reset login on dropdown hide
    }

    // Print


    _createClass(Utils, [{
        key: 'print',
        value: function print() {
            console.log('I am Utils!');
        }

        // Build a Form

    }, {
        key: 'buildForm',
        value: function buildForm(keys) {
            var args = Array.prototype.slice.call(arguments, 1);
            var form = {};
            keys.forEach(function (key, index) {
                form[key] = args[index];
            });
            return form;
        }

        // Serialize a Form

    }, {
        key: 'serializeForm',
        value: function serializeForm(id) {
            var form = $(id).serializeArray();
            return form;
        }

        // Unique values

    }, {
        key: 'unique',
        value: function unique(data) {
            return new Set(data);
        }

        // Scroll to div

    }, {
        key: 'scrollTo',
        value: function scrollTo(location) {
            if (location !== undefined) {
                var scrolldiv = $(location);
                $('html,body').animate({ scrollTop: scrolldiv.offset().top }, 1500, 'easeInOutExpo');
            } else {
                $('html,body').animate({ scrollTop: 0 }, 1500, 'easeInOutExpo');
            }
        }

        // Initialize Info Pop-Overs

    }, {
        key: 'initInfoPopOvers',
        value: function initInfoPopOvers() {
            $('.infopop [data-toggle="popover"]').popover();
        }

        // Initialize tooltips

    }, {
        key: 'initToolTips',
        value: function initToolTips() {
            $('[data-toggle="tooltip"]').tooltip();
        }

        // Select Choices from a Bootstrap-Select element

    }, {
        key: 'selectChoices',
        value: function selectChoices(id, choices) {
            $(id).selectpicker('val', choices);
            $(id).selectpicker('refresh');
        }

        // Reset Choices from a Bootstrap-Select element

    }, {
        key: 'resetChoices',
        value: function resetChoices(id) {
            console.log('reseting in utils', id);
            var select = typeof id === 'string' ? $(id) : id;
            select.selectpicker('deselectAll');
            select.selectpicker('refresh');
            select.selectpicker('render');
        }

        // Login function

    }, {
        key: 'login',
        value: function login() {
            var _this2 = this;

            var form = $('#loginform').serialize();
            Promise.resolve($.post(Flask.url_for('index_page.login'), form, 'json')).then(function (data) {
                if (data.result.status < 0) {
                    throw new Error('Bad status login. ' + data.result.message);
                }
                if (data.result.message !== '') {
                    var stat = data.result.status === 0 ? 'danger' : 'success';
                    var htmlstr = '<div class=\'alert alert-' + stat + '\' role=\'alert\'><h4>' + data.result.message + '</h4></div>';
                    $('#loginmessage').html(htmlstr);
                }
                if (data.result.status === 1) {
                    location.reload(true);
                }
            }).catch(function (error) {
                _this2.resetLogin();
                alert('Bad login attempt! ' + error);
            });
        }

        // Reset Login

    }, {
        key: 'resetLogin',
        value: function resetLogin() {
            console.log('reset');
            $('#loginform').trigger('reset');
            $('#loginmessage').empty();
        }

        // Submit Login on Keyups

    }, {
        key: 'submitLogin',
        value: function submitLogin(event) {
            var _this = event.data;
            // login
            if (event.keyCode == 13) {
                if ($('#login-user').val() && $('#login-pass').val()) {
                    _this.login();
                }
            }
        }

        // Shows a banner

    }, {
        key: 'marvinBanner',
        value: function marvinBanner(text, expiryDays, cookieName, url, urlText) {

            var _this = this;
            expiryDays = expiryDays === undefined ? 0 : expiryDays;
            cookieName = cookieName === undefined ? "marvin_banner_cookie" : cookieName;
            url = url === undefined ? "" : url;
            urlText = urlText === undefined ? "Learn more" : urlText;

            if (urlText === "" || url === "") {
                urlText = "";
                url = "";
            }

            _this.window[0].cookieconsent.initialise({
                "palette": {
                    "popup": {
                        "background": "#000"
                    },
                    "button": {
                        "background": "#f1d600"
                    }
                },
                "position": "top",
                "cookie": {
                    "name": cookieName,
                    "expiryDays": expiryDays,
                    "domain": "localhost" },
                "content": {
                    "message": text,
                    "dismiss": 'Got it!',
                    "href": url,
                    "link": urlText }
            });

            if (expiryDays === 0) {
                document.cookie = cookieName + '=;expires=Thu, 01 Jan 1970 00:00:01 GMT;path=/;domain=localhost';
            }
        }
    }]);

    return Utils;
}();
