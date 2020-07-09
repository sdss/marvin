/*
 * Filename: explore.js
 * Project: marvin
 * Author: Brian Cherinka
 * Created: Thursday, 9th July 2020 10:09:14 am
 * License: BSD 3-clause "New" or "Revised" License
 * Copyright (c) 2020 Brian Cherinka
 * Last Modified: Thursday, 9th July 2020 2:34:54 pm
 * Modified By: Brian Cherinka
 */

//
// Javascript Explore object handling JS things for exploring batch set of galaxies
//
//jshint esversion: 6
'use strict';

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
        }
    }]);

    return Explore;
}();
