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
            });
        }
    }]);

    return Explore;
}();
