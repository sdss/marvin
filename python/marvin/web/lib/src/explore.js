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


class Explore {

    // Constructor
    constructor(targets) {
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
    print() {
        console.log('We are now printing explore', this.targets);
    }

    // Fetch a map and initialize the DAP heatmap display
    get_map(input) {
        let _this = this;
        let param = this.mapparam.prop('value');
        let bintemp = this.bintemp.prop('value');

        let [target, div] = input;
        let mapdiv = $(div).find('div').first();
        mapdiv.empty();

        let data = {'body': JSON.stringify({'target': target, 
            'mapchoice': param, 'btchoice': bintemp}), 
            'method':'POST', 'headers':{'Content-Type':'application/json'}};

        return fetch(Flask.url_for('explore_page.webmap'), data)
        .then((response) => response.json())
        .then((data) => {
            if (data.result.maps !== undefined && data.result.maps.data !== null) {
                this.heatmap = new HeatMap(mapdiv, data.result.maps.data, data.result.maps.msg,
                    data.result.maps.plotparams, null);
                this.heatmap.mapdiv.highcharts().reflow();
            } else {
                let err = `<p class='alert alert-danger'>${data.result.msg}</p>`;
                mapdiv.html(err);
            }
            return "done";
        });
    }

    // Grab all map data for list of targets
    get_maps(event) {
        let _this = event.data;

        // set button to loading...
        $(this).button('loading');

        // get the target list
        let targets = _this.targets;

        // construct the input data
        const zip = (a, b) => a.map((k, i) => [k, b[i]]);
        let data = zip(targets, _this.mapsdiv.children('div'));

        // create a list of promises to run
        Promise.allSettled(data.map((d) => {
            return _this.get_map(d);
        }))
        .then(response=>{
            console.log("all done");
            return _this.mapsbtn.button('reset');
        });

    }

}