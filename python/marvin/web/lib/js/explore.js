/*
 * Filename: explore.js
 * Project: marvin
 * Author: Brian Cherinka
 * Created: Thursday, 9th July 2020 10:09:14 am
 * License: BSD 3-clause "New" or "Revised" License
 * Copyright (c) 2020 Brian Cherinka
 * Last Modified: Friday, 10th July 2020 1:40:00 pm
 * Modified By: Brian Cherinka
 */

//
// Javascript Explore object handling JS things for exploring batch set of galaxies
//
//jshint esversion: 6
'use strict';

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Explore = function () {

    // Constructor
    function Explore(targets) {
        _classCallCheck(this, Explore);

        this.targets = targets;
        this.explorediv = $('#explorediv');
        this.mapsdiv = this.explorediv.find('#exmaps');
        this.mapdiv = this.explorediv.find('#exmapdiv1');
        this.mapsbtn = $('#getmapbut');

        this.mapparam = $('#mapchoice');
        this.bintemp = $('#btchoice');

        this.mapsbtn.on('click', this, this.testall); // this event fires when a user clicks the MapSpec View Tab
    }

    // Test print


    _createClass(Explore, [{
        key: 'print',
        value: function print() {
            console.log('We are now printing explore', this.targets);
        }

        // Initialize the DAP Heatmap displays

    }, {
        key: 'initHeatmap',
        value: function initHeatmap(maps, mapmsgs) {
            var start = performance.now();
            var mapchildren = this.mapsdiv.children('div');
            var _this = this;
            $.each(mapchildren, function (index, child) {
                var mapdiv = $(child).find('div').first();
                mapdiv.empty();
                if (maps[index] !== undefined && maps[index].data !== null) {
                    this.heatmap = new HeatMap(mapdiv, maps[index].data, maps[index].msg, maps[index].plotparams, _this);
                    this.heatmap.mapdiv.highcharts().reflow();
                } else {
                    var err = '<p class=\'alert alert-danger\'>' + mapmsgs[index] + '</p>';
                    mapdiv.html(err);
                }
            });
            var end = performance.now();
            console.log('td', end - start, 'in ms');
        }

        // Init a single Heatmap

    }, {
        key: 'initSingleHeatmap',
        value: function initSingleHeatmap(data) {
            var _this = this;
            //let mapdiv = $(child).find('div').first();
            //mapdiv.empty();

            var _data = _slicedToArray(data, 3),
                mapobj = _data[0],
                mapmsg = _data[1],
                div = _data[2];

            var mapdiv = $(div).find('div').first();
            mapdiv.empty();
            if (mapobj !== undefined && mapobj.data !== null) {
                this.heatmap = new HeatMap(mapdiv, mapobj.data, mapobj.msg, mapobj.plotparams, _this);
                this.heatmap.mapdiv.highcharts().reflow();
            } else {
                var err = '<p class=\'alert alert-danger\'>' + mapmsg + '</p>';
                mapdiv.html(err);
            }
        }

        // Parallel process heatmaps

    }, {
        key: 'parallelHeatmaps',
        value: function parallelHeatmaps(maps, mapmsgs) {
            var zip = function zip(a, b, c) {
                return a.map(function (k, i) {
                    return [k, b[i], c[i]];
                });
            };
            var mapdivs = this.mapsdiv.children('div'); //.find("div");
            var data = zip(maps, mapmsgs, mapdivs);
            console.log(data);
            var p = new Parallel(data);
            p.map(this.initSingleHeatmap);
            //this.initSingleHeatmap(data[0]);
        }
    }, {
        key: 'promise',
        value: function promise(maps, mapmsgs) {
            this.mapsbtn.prop('disabled', true);
            var start = performance.now();
            var _this = this;
            var zip = function zip(a, b, c) {
                return a.map(function (k, i) {
                    return [k, b[i], c[i]];
                });
            };
            var mapdivs = this.mapsdiv.children('div'); //.find("div");
            var data = zip(maps, mapmsgs, mapdivs);

            Promise.all(data.map(function (id) {
                //console.log('id', id);
                _this.initSingleHeatmap(id);
                return "";
            })).then(function (results) {
                // results is an array of names
                console.log('res', results);
                var end = performance.now();
                console.log('td', end - start, 'in ms');
                _this.mapsbtn.prop('disabled', false);
                _this.mapsbtn.button('reset');
            });
        }
    }, {
        key: 'test',
        value: function test(input) {
            var _this2 = this;

            var _this = this;
            var param = this.mapparam.prop('value');
            var bintemp = this.bintemp.prop('value');

            var _input = _slicedToArray(input, 2),
                target = _input[0],
                div = _input[1];

            var mapdiv = $(div).find('div').first();
            mapdiv.empty();

            var data = { 'body': JSON.stringify({ 'target': target,
                    'mapchoice': param, 'btchoice': bintemp }),
                'method': 'POST', 'headers': { 'Content-Type': 'application/json' } };

            //let mapdivs = this.mapsdiv.children('div');
            //let mapdiv = mapdivs.find('div').first();
            //mapdiv.empty();
            //console.log('test', target, mapdiv);

            fetch(Flask.url_for('explore_page.webmap'), data).then(function (response) {
                return response.json();
            }).then(function (data) {
                //console.log('json', data.result);
                if (data.result.maps !== undefined && data.result.maps.data !== null) {
                    _this2.heatmap = new HeatMap(mapdiv, data.result.maps.data, data.result.maps.msg, data.result.maps.plotparams, _this);
                    _this2.heatmap.mapdiv.highcharts().reflow();
                } else {
                    var err = '<p class=\'alert alert-danger\'>' + data.result.msg + '</p>';
                    mapdiv.html(err);
                }
            });
            //fetch('http://localhost:5000/marvin/explore/webmap/', {'body': JSON.stringify({'release':'DR17', 'target':'8485-1901', 'mapchoice':'emline_gflux_ha_6564', 'btchoice':'HYB10-MILESHC-MASTARSSP'}), 'method':'POST', 'headers':{'Content-Type':'application/json'}}).then(response=>response.json()).then(data => console.log(data));
        }
    }, {
        key: 'testall',
        value: function testall(event) {
            var _this = event.data;
            // if (targets === undefined) {
            //     targets = this.targets;
            // }
            var targets = _this.targets;

            var start = performance.now();
            //let _this = this;
            //_this.mapsdiv.children('div').empty();
            var zip = function zip(a, b) {
                return a.map(function (k, i) {
                    return [k, b[i]];
                });
            };
            var data = zip(targets, _this.mapsdiv.children('div'));
            Promise.all(data.map(function (d) {
                //let mapdiv = _this.mapsdiv.children('div').find('div[id^="exmapdiv"]')[index];
                //console.log(mapdiv);
                _this.test(d);
            })).then(function (response) {
                var end = performance.now();
                console.log('td', end - start, 'in ms');
                _this.mapsbtn.prop('disabled', false);
                _this.mapsbtn.button('reset');
                console.log("all done");
            });
        }
    }]);

    return Explore;
}();
