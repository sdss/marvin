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

        // event handlers
        this.mapsbtn.on('click', this, this.get_maps); // this event fires when a user clicks the Get Maps button
    }

    // Test print


    _createClass(Explore, [{
        key: 'print',
        value: function print() {
            console.log('We are now printing explore', this.targets);
        }

        // Fetch a map and initialize the DAP heatmap display

    }, {
        key: 'get_map',
        value: function get_map(input) {
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

            return fetch(Flask.url_for('explore_page.webmap'), data).then(function (response) {
                return response.json();
            }).then(function (data) {
                if (data.result.maps !== undefined && data.result.maps.data !== null) {
                    _this2.heatmap = new HeatMap(mapdiv, data.result.maps.data, data.result.maps.msg, data.result.maps.plotparams, _this);
                    _this2.heatmap.mapdiv.highcharts().reflow();
                } else {
                    var err = '<p class=\'alert alert-danger\'>' + data.result.msg + '</p>';
                    mapdiv.html(err);
                }
                return "done";
            });
        }

        // Grab all map data for list of targets

    }, {
        key: 'get_maps',
        value: function get_maps(event) {
            var _this = event.data;

            // set button to loading...
            $(this).button('loading');

            // get the target list
            var targets = _this.targets;

            // construct the input data
            var zip = function zip(a, b) {
                return a.map(function (k, i) {
                    return [k, b[i]];
                });
            };
            var data = zip(targets, _this.mapsdiv.children('div'));

            // create a list of promises to run
            Promise.allSettled(data.map(function (d) {
                return _this.get_map(d);
            })).then(function (response) {
                console.log("all done");
                return _this.mapsbtn.button('reset');
            });
        }
    }]);

    return Explore;
}();
