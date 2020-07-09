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


class Explore {

    // Constructor
    constructor(targets) {
        this.targets = targets;
        this.explorediv = $('#explorediv');
        this.mapsdiv = this.explorediv.find('#exmaps');
        this.mapdiv = this.explorediv.find('#exmapdiv1');


    }

    // Test print
    print() {
        console.log('We are now printing explore', this.targets);
    }

    // Initialize the DAP Heatmap displays
    initHeatmap(maps, mapmsgs) {
        let mapchildren = this.mapsdiv.children('div');
        let _this = this;
        $.each(mapchildren, function (index, child) {
            let mapdiv = $(child).find('div').first();
            mapdiv.empty();
            if (maps[index] !== undefined && maps[index].data !== null) {
                this.heatmap = new HeatMap(mapdiv, maps[index].data, maps[index].msg,
                    maps[index].plotparams, _this);
                this.heatmap.mapdiv.highcharts().reflow();
            } else {
                let err = `<p class='alert alert-danger'>${mapmsgs[index]}</p>`;
                mapdiv.html(err);
            }
        });
    }

}